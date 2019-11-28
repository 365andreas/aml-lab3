from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from keras.callbacks import ModelCheckpoint
from biosppy.signals import ecg
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import pandas as pd
import scipy.io as sio
from os import listdir
from os.path import isfile, join
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM, Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras import regularizers
from keras import backend as K

from sklearn.impute import SimpleImputer
from copy import deepcopy

np.random.seed(7)

number_of_classes = 4 #Total number of classes

def change(x):  #From boolean arrays to decimal arrays
    answer = np.zeros((np.shape(x)[0]))
    for i in range(np.shape(x)[0]):
        max_value = max(x[i, :])
        max_index = list(x[i, :]).index(max_value)
        answer[i] = max_index
    return answer.astype(np.int)

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

Y = pd.read_csv('y_train.csv')
Y = Y.y
Y_raw = deepcopy(Y)
Y = Y.to_numpy()
size = Y.shape[0]
print('size: ', size)
#Train_data = pd.read_csv(mypath + 'REFERENCE.csv', sep=',', header=None, names=None)
X = pd.read_csv('X_train.csv', header=0)
X_raw = deepcopy(X)

# # feat_size=X.shape[1]
# imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# X = imputer.fit_transform(X)

# Label_set = np.zeros((size, number_of_classes))

# # binary representation of the 4 classes
# for i in range(size):
#     dummy = np.zeros((number_of_classes))
#     dummy[int(Y[i])] = 1
#     Label_set[i, :] = dummy

# X = (X - X.mean())/(X.std())  #Some normalization here
# X = np.expand_dims(X, axis=2) #For Keras' data input size - add other dimension

# values = [i for i in range(size)] #shuffle
# permutations = np.random.permutation(values)
# X = X[permutations, :]
# Label_set = Label_set[permutations, :]


# print(X.shape)
# print(Label_set.shape)


# train = 0.9 #Size of training set in percentage
# X_train = X[:int(train * size), :]
# Y_train = Label_set[:int(train * size), :]
# X_val = X[int(train * size):, :]
# Y_val = Label_set[int(train * size):, :]

def create_model(feat_size):
    model = Sequential()
    model.add(Conv1D(128, 55, activation='relu', input_shape=(feat_size, 1)))
    model.add(MaxPooling1D(10))
    model.add(Dropout(0.5))
    model.add(Conv1D(128, 25, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.5))
    model.add(Conv1D(128, 10, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.5))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalAveragePooling1D())
    # model.add(Flatten())
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, kernel_initializer='normal', activation='softmax'))
    return model

def train_and_evaluate__model(model, X_train, Y_train, X_val, Y_val, ep):

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])
    checkpointer = [ModelCheckpoint(filepath='conv_models/best_model.h5', monitor='val_accuracy', verbose=1,
                                        save_best_only=True, mode='max')]
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=275, epochs=ep, verbose=2, shuffle=True,
              callbacks=checkpointer)

    dependencies = {
     'f1_m': f1_m
    }
    model_best = load_model('conv_models/best_model.h5', custom_objects=dependencies)

    return model_best

val_scores = []

x_train_init = X_raw
y_train_init = Y_raw
x_train = np.asarray(x_train_init)
y_train = np.asarray(y_train_init)

N = 2
epochs = 2
kf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)
for train_index, test_index in kf.split(x_train, y_train):
    X_train_i, X_test_i = x_train_init.iloc[train_index], x_train_init.iloc[test_index]
    Y_train_i, Y_test_i = y_train_init.iloc[train_index], y_train_init.iloc[test_index]


    imputer   = SimpleImputer(missing_values=np.nan, strategy='median')
    X_train_i = imputer.fit_transform(X_train_i)
    feat_size = X_train_i.shape[1]

    X_train_i = (X_train_i - X_train_i.mean())/(X_train_i.std())
    X_train_i = np.expand_dims(X_train_i, axis=2)

    X_train_i = np.asarray(X_train_i)
    Y_train_i = np.asarray(Y_train_i)

    # Y_train_i = Y_train_i.y
    # Y_raw = deepcopy(Y_train_i)
    # Y_train_i = Y_train_i.to_numpy()
    size = Y_train_i.shape[0]
    print('size: ', size)

    Label_set = np.zeros((size, number_of_classes))

    # binary representation of the 4 classes
    for i in range(size):
        dummy = np.zeros((number_of_classes))
        dummy[int(Y[i])] = 1
        Label_set[i, :] = dummy

    train = 0.9
    X_train = X_train_i[:int(train * size), :]
    Y_train = Label_set[:int(train * size), :]
    X_val   = X_train_i[int(train * size):, :]
    Y_val   = Label_set[int(train * size):, :]

    model = None
    model = create_model(feat_size)

    best_model = train_and_evaluate__model(model, X_train, Y_train, X_val, Y_val, epochs)

    X_test_i  = np.asarray(X_test_i)
    Y_test_i  = np.asarray(Y_test_i)

    X_test = imputer.transform(X_test_i)

    X_test = (X_test - X_test.mean())/(X_test.std())  # Some normalization here
    X_test = np.expand_dims(X_test, axis=2)           # For Keras' data input size - add other dimension

    predictions = best_model.predict(X_test)
    score = f1_score(Y_test_i, change(predictions), average='micro')

    val_scores.append(score)

print(val_scores)
print("mean: ", np.mean(val_scores))
print("std: ", np.std(val_scores))

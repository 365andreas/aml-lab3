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

target_train = pd.read_csv('y_train.csv')
target_train = target_train.y
target_train = target_train.to_numpy()
size = target_train.shape[0]
print('size: ', size)
#Train_data = pd.read_csv(mypath + 'REFERENCE.csv', sep=',', header=None, names=None)
X = pd.read_csv('X_train.csv', header=0)
feat_size=X.shape[1]
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
X = imputer.fit_transform(X)

Label_set = np.zeros((size, number_of_classes))

# binary representation of the 4 classes
for i in range(size):
    dummy = np.zeros((number_of_classes))
    dummy[int(target_train[i])] = 1
    Label_set[i, :] = dummy

X = (X - X.mean())/(X.std())  #Some normalization here
X = np.expand_dims(X, axis=2) #For Keras' data input size - add other dimension

values = [i for i in range(size)] #shuffle
permutations = np.random.permutation(values)
X = X[permutations, :]
Label_set = Label_set[permutations, :]


print(X.shape)
print(Label_set.shape)


train = 0.9 #Size of training set in percentage
X_train = X[:int(train * size), :]
Y_train = Label_set[:int(train * size), :]
X_val = X[int(train * size):, :]
Y_val = Label_set[int(train * size):, :]

# def train_and_evaluate__model(model, X_train, Y_train, X_val, Y_val, i):

ep = 5
# def create_model():
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
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m])
checkpointer = ModelCheckpoint(filepath='conv_models/best_model.h5', monitor='val_accuracy', verbose=1,
                               save_best_only=True, mode='max')
hist = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=275, epochs=ep, verbose=2, shuffle=True,
                 callbacks=[checkpointer])
pd.DataFrame(hist.history).to_csv(path_or_buf='conv_models/History.csv')
predictions = model.predict(X_val)
score = accuracy_score(change(Y_val), change(predictions))
score_f1 = f1_score(change(Y_val), change(predictions), average='micro')
print('Last epoch\'s validation score is ', score, ' and f1_score: ', score_f1)
df = pd.DataFrame(change(predictions))
df.to_csv(path_or_buf='conv_models/Preds_' + str(format(score, '.4f')) + '.csv', index=None, header=None)
pd.DataFrame(confusion_matrix(change(Y_val), change(predictions))).to_csv(path_or_buf='conv_models/Result_Conf' + str(format(score, '.4f')) + '.csv', index=None, header=None)

# 5. Make predictions
dependencies = {
     'f1_m': f1_m
}
model = load_model('conv_models/best_model.h5', custom_objects=dependencies)

x_test = pd.read_csv("X_test.csv")

x_test = imputer.transform(x_test)

x_test = (x_test - x_test.mean())/(x_test.std())  # Some normalization here
x_test = np.expand_dims(x_test, axis=2)           # For Keras' data input size - add other dimension

y_test = model.predict_classes(x_test)
y_test = pd.DataFrame(y_test, columns=['y'])
df = pd.DataFrame(list(range(0, 3411)), columns=['id'])
df.insert(1, "y", y_test)
df.to_csv('solution.csv', index=False)

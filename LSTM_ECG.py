from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

# Embedding
max_features = 20000
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 2

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

print('Loading data...')
X = pd.read_csv('X_train.csv', header=0)
X = X.drop('id', axis=1)

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

minimum = np.floor(np.amin(X))

#X ranges from 0 to new maximum
X = X + abs(minimum)
maximum = np.amax(X)

#X ranges from 0 to max_features(= 20000) [0, 20000)
X = X * (max_features - 1) / maximum
X = X.astype(int)


# X = X.to_numpy()
feat_size = X.shape[1]

Y = pd.read_csv('y_train.csv')
Y = Y.y
Y = Y.to_numpy()
no_of_samples = Y.shape[0]
print('no_of_samples: ', no_of_samples)

number_of_classes = 4
Label_set = np.zeros((no_of_samples, number_of_classes))
# binary representation of the 4 classes
for i in range(no_of_samples):
    dummy = np.zeros((number_of_classes))
    dummy[int(Y[i])] = 1
    Label_set[i, :] = dummy

Y = Label_set

train = 0.9 # Size of training set in percentage
x_train = X[                           : int(train * no_of_samples), :]
y_train = Y[                           : int(train * no_of_samples)]
x_val =   X[int(train * no_of_samples) :                           , :]
y_val =   Y[int(train * no_of_samples) :                           ]

print(len(x_train), 'train sequences')
print(len(x_val), 'test sequences')

print('Build model...')

model = Sequential()
model.add(Embedding(max_features , embedding_size, input_length=feat_size))
model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_val, y_val),
          verbose=1)
score, acc = model.evaluate(x_val, y_val, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

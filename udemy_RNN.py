# one -> one ; one -> many; many -> one
#sunspring movie 
#vanishing/exploding gradient problem; small weigth -> weight update slow 
#LSTM 

#google stock price trend example
import imp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler

# data preprocessing
dataset_train = pd.read('google_stock.csv')
training_set = dataset_train.iloc[:, 1:2].values #upper bound excluded -> numpy array
#apply feature scaling, standardisation, normalisation
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled  = sc.fit_transform(training_set)

#data strcuture with 60 timesteps (tested), 1 output
#x`; 0-59 day; y: 60 day; last 60 values -> new 
X_train, y_train = [], []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#reshape add a dimension, be comparable to the input shape
#batch step, timestep, input_dim
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Build
regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2)) #20% drop-out
regressor.add(LSTM(units=50, return_sequences=True)) 
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1)) #predictor

#Train
regressor.compile(optimizer='Adam', loss="mean_squared_error") #regression 

#fit
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

#make predcitions
#get 60
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) #reshape into correct 3D structure
predict = regressor.predict(X_test)
predict = sc.inverse_transform(predict) #back to normal scale


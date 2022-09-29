#take results from SOM, combine with supervised modles -> output possibility of eery person who cheats
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from minisom import MiniSom

dataset = pd.read_csv('credit_card.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

#feature scaling
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)
#som 
som = MiniSom(10,10, input_len = 15, sigma=1.0, learning_rate=0.5)
#random inti weight
som.andom_weights_init(X)
som.train_random(data = X, num_iteration = 100)

#visualize
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(X)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], markeredgecolor = colors[y[i]], markgerfacecolor='None',
    markersize=10, markeredgewidh=2)
show()

mappings = som.win_map(X)
frauds = np.concatenate((mappings[{8,1}], mappings[(6,8)]), axis=0) #vertical line
frauds = sc.inverse_transform(frauds)

####output from SOM
frauds = np.concatenate((mappings[{5,3}], mappings[(8,3)]), axis=0)  #got from the output map generated
frauds = sc.inverse_transform(frauds)

#create dependant variable 
custmors = dataset.iloc[:, 1:].values
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[:,0] in frauds:
        is_fraud[i] = 1


# Feature Scaling
# for NN, scaling is necessary ,just scale everything
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
custmors = sc.fit_transform(custmors)

classifier = Sequential()
classifier.add(Dense(units=2, kernel_initializer="uniform", activation='relu', input_dim=15))
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(custmors, is_fraud, batch_size=1, epochs=2)
y_pred = classifier.predict(custmors)
y_pred = np.concatenate((dataset.iloc[:, 0].values,y_pred), axis=1)
y_pred = y_pred[y_pred[:,2].argsort()]


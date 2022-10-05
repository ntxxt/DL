#reduce dimensionality, unsupervised learning
# k-means
#choose cluster => select random k points -> assgin each point to nearest centroid -> relocate new centroid -> until not change
#weigts in SOM: are charactistics of the map themselves; distance is shown as euclidean distance
#actual points near the centra, will have weights being updated closer to the center
#pull -> shink
#SOM retain topology of data; reveal relations that are not easily identified; learns without superision; no backpropagation
#no lateral connections between output nodes; 
#k-means++ random selection of centroids at first will not affect the result
#select the correct number of centriods; WCSS, centriods increase, better fit; pick where the changes happen;

#case: if the customer cheats or not
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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






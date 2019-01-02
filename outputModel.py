import numpy as np
from kmodes.kmodes import KModes
import csv


total_clusters = 12
lst = []

data = np.genfromtxt('data.csv',delimiter=',')

km = KModes(n_clusters=total_clusters, init='Huang', n_init=5, verbose=0)

clusters = km.fit_predict(data)

# Print the cluster centroids
lst.append(km.cluster_centroids_)
print(lst)

with open('output.csv', 'wb') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(lst)

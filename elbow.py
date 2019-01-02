import numpy as np
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt


# categorical data
data = np.genfromtxt('data.csv',delimiter=',')
lst = []
n = 20 #user change for elbow method testing

def process(n):
    km = KModes(n_clusters=n, init='Huang', n_init=5, verbose=0)
    clusters = km.fit_predict(data)
    lst.append(km.cost_)

for i in range(n):
    process(i+1)

print(lst)
plt.plot(range(1,n+1),lst)
plt.show()

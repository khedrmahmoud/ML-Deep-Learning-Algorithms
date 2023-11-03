
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("Mall_Customers.csv")

x = dataset.iloc[:,[3,4]].values
# y = dataset.iloc[:, -1].values
#%%
"""# Using the dendrogram method to find the optimal number of clusters"""
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method='ward')) 
plt.title("Dendrograme")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

#%%
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage='ward')
y_hc = hc.fit_predict(x)

#%%
"""Visualizing the clusters
"""

plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'violet', label = 'Cluster4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'yellow', label = 'Cluster5')


plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

plt.show()
# coding: utf-8
########################################################################################################################
## PCA example with Iris Data-set
########################################################################################################################
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
warnings.filterwarnings("ignore", category=DeprecationWarning)

np.random.seed(5)

## original data
iris = datasets.load_iris()
X = iris.data
y = iris.target

## transformed data
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

## plot
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:												# Define o label de cada eixo
	ax.text3D(X[y == label, 0].mean(), X[y == label, 1].mean() + 1.5, X[y == label, 2].mean(),
				name,
				horizontalalignment='center',
				bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

## Reorder the labels to have colors matching the y (i.e iris.target)
y = np.choose(y, [1, 2, 0]).astype(np.float)

## plot
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral)
plt.show()

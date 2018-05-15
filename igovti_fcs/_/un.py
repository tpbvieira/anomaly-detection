from sklearn import cluster, datasets

iris = datasets.load_iris()
x_iris = iris.data
y_iris = iris.target

k_means = cluster.KMeans(n_clusters=3)
k_means.fit(x_iris) 

print(iris)
print(k_means.labels_[::10])
print(y_iris[::10])

from __future__ import print_function
import fbpca
import numpy as np
import numpy.random
import time
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd
r = int(1e5)
c = int(1.5e3)
start = time.time()
A = np.random.uniform(size=[r, c])
end = time.time()
print("Done generating test matrix, it took %2.2f seconds." % (end - start))

print("Running pca on matrix of size %d" % (r * c))

for k in [2, 10, 20, 50, 100]:    
    print("Using k=", k)

    # fbpca
    start = time.time()
    (U, s, Va) = fbpca.pca(A, k=k, raw=True, n_iter=2)
    end = time.time()
    print("\tfbpca: %2.2f" % (end - start))

    # sklearn.pca
    start = time.time()
    sklearn_pca = PCA(k, copy=False, whiten=False, svd_solver='randomized', iterated_power=2)
    sklearn_pca.fit(A)
    end = time.time()
    print("\tsklearn.pca: %2.2f" % (end - start))

    # sklearn.randomized_svd
    start = time.time()
    randomized_svd(A, k, n_iter=2)    
    end = time.time()
    print("\tsklearn.randomized_svd: %2.2f" % (end - start))
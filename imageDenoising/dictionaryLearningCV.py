from time import time

import os.path
import sys
import numpy as np
import scipy as sp
import pandas as pd  																									# data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version

from skimage.measure import (compare_mse, compare_nrmse, compare_psnr)


# settings
filePath = "/media/thiago/ubuntu/datasets/fraudDetection/"


# Function to plot image difference
def print_comparison(target, reference):
	difference = target - reference
	mse = compare_mse(reference, target)
	nrmse = compare_nrmse(reference, target)
	psnr = 0
	text = 'norm: %(norm).4f\tMSE: %(MSE).4fs\tNRMSE: %(NRMSE).4fs\tPSNR: %(PSNR).4fs' % {'norm': np.sqrt(np.sum(difference ** 2)), 'MSE': mse, 'NRMSE': nrmse, 'PSNR': psnr}
	print(text)


print('\n## Loading saved data...')
# data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/train_under_data.csv', index_col=0)
data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/train_data.csv', index_col=0)
print('Data: ' + str(data.shape))


L = data.shape[0]
N = data.shape[1]
K_range = [10, 50, 100, 200, 500]  # dictionary samples
noIt_range = [10, 50, 100]
tnz_range = [1, 2, 3, 5, 7, 10]


for tnz in tnz_range:
	for K in K_range:
		for noIt in noIt_range:

			transform_algorithms = [
				('MiniBatchDL OMP', 'omp', {'transform_n_nonzero_coefs': tnz})
				# ,('Orthogonal Matching Pursuit (Sparsity: 5)', 'omp', {'transform_n_nonzero_coefs': 5})
				# ,('Least-angle regression (5 atoms)', 'lars', {'transform_n_nonzero_coefs': 5})
				# ,('Thresholding (alpha=0.1)', 'threshold', {'transform_alpha': .1})
				# ,('RLS-DLA (Sparsity: 2)', '', {})
				# ,('RLS-DLA (Sparsity: 5)', '', {})
				# ,('K-SVD (Sparsity: 2)', '', {})
				# ,('T-MOD', '', {})
				# ,('K-HOSVD (Sparsity: 2)', '', {})
				# ,('MOD (Sparsity: 5)', '', {})
				# ,('MOD (Sparsity: 2)', '', {})
				]

			for title, transform_algorithm, kwargs in transform_algorithms:
				print("\n"+title + ':')
				t0 = time()
				dictionary = {};
				if title is 'MOD (Sparsity: 2)':
					dictMiniBatch = np.loadtxt(filePath + 'dictMODNoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=2.csv', delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeMODRefNoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=2.csv', delimiter=';')
				elif title is 'MOD (Sparsity: 5)':
					dictMiniBatch = np.loadtxt(filePath + 'dictMODNoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=5.csv', delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeMODRefNoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=5.csv', delimiter=';')
				elif title is 'RLS-DLA (Sparsity: 2)':
					dictMiniBatch = np.loadtxt(filePath + 'dictRLS-DLANoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=2.csv', delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeRLS-DLARefNoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=2.csv', delimiter=';')
				elif title is 'RLS-DLA (Sparsity: 5)':
					dictMiniBatch = np.loadtxt(filePath + 'dictRLS-DLANoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=5.csv', delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeRLS-DLARefNoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=5.csv', delimiter=';')
				elif title is 'K-SVD (Sparsity: 2)':
					dictMiniBatch = np.loadtxt(filePath + 'dictK-SVDNoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=2.csv', delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeK-SVDRefNoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=2.csv', delimiter=';')
				elif title is 'T-MOD (Sparsity: 2)':
					continue
				elif title is 'K-HOSVD (Sparsity: 2)':
					dictMiniBatch = np.loadtxt(filePath + 'dictK-HOSVDNoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=2.csv', delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeK-HOSVDRefNoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=2.csv', delimiter=';')
				else:  # Learning MiniBatch Dictionary and Orthogonal Matching Pursuit (Sparsity: 2)
					t0 = time()
					miniBatch = MiniBatchDictionaryLearning(n_components=K, alpha=1, n_iter=noIt)
					miniBatchDict = miniBatch.fit(data.values).components_
					dictionary = miniBatchDict;
					dt = time() - t0
					# print('miniBatchDict: ' + str(miniBatchDict.shape))
					print('Fit Time: %.2fs.' % dt)
					print('## L={:d}_N={:d}_tnz={:d}_K={:d}_noIt={:d}'.format(L, N, tnz, K, noIt))
					t0 = time()
					miniBatch.set_params(transform_algorithm=transform_algorithm, **kwargs)
					sparseCode = miniBatch.transform(data)
				reconstruction = np.dot(sparseCode, dictionary)
				reconstruction = pd.DataFrame(reconstruction, index=data.index.values)
				dt = time() - t0
				# print('SparseCode: ' + str(sparseCode.shape))
				# print('Reconstruction: ' + str(reconstruction.shape))
				print('Transform Time: %.2fs.' % dt)
				print_comparison(reconstruction.values, data.values)

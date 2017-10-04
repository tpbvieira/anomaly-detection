# coding: utf-8
####################################################################################################################################################################
# ## Cross-Validation for evaluating the reconstruction performance by dictionary learning methods and parameter configurations applied to a fraud detection dataset
####################################################################################################################################################################

import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import matlab.engine
from time import time
from sklearn.decomposition import MiniBatchDictionaryLearning
from skimage.measure import (compare_mse, compare_nrmse)

## settings
filePath = "/media/thiago/ubuntu/datasets/fraudDetection/"
results = open('results/fraudDetectionDLCV.txt', 'w')
# matlab = matlab.engine.start_matlab()

## Function to plot image difference
def print_comparison(target, reference):
	difference = target - reference
	mse = compare_mse(reference, target)
	nrmse = compare_nrmse(reference, target)
	psnr = 0
	text = 'norm:%(norm).4f\tMSE:%(MSE).4fs\tNRMSE:%(NRMSE).4fs\tPSNR:%(PSNR).4fs' % {'norm': np.sqrt(np.sum(difference ** 2)), 'MSE': mse, 'NRMSE': nrmse, 'PSNR': psnr}
	print >> results, text

## Load data
print >> results, '\n## Loading saved data...'
# data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/train_under_data.csv', index_col=0)
data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/train_data.csv', index_col=0)
# print >> results, 'Data Shape: ' + str(data.shape)
# matlab.cd(r'/home/thiago/dev/projects/anomaly-detection/distributed-tensor-dictionary-learning', nargout=0)
# matlab.project(nargout=0)

## parameters
L = data.shape[0]
N = data.shape[1]
# K_range = [10, 50, 100, 200, 500]
# noIt_range = [10, 50, 100]
# tnz_range = [2, 3, 5, 7]
K_range = [100]
noIt_range = [50]
tnz_range = [5]

## Load previously calculated data and evaluates their reconstruction performance for each parameter configuration
## ToDo: Evaluate differents sparse coding solvers, such as OMP, Lasso, JavaOMP and others
for K in K_range:
	for tnz in tnz_range:
		for noIt in noIt_range:
			fileNameSufix = 'L={:d}_K={:d}_noIt={:d}_solver=javaORMP_tnz={:d}.csv'.format(L, N, tnz, K, noIt)
			transform_algorithms = [
				('MiniBatchDL OMP', 'omp', {'transform_n_nonzero_coefs': tnz})
				, ('MOD', '', {})
				, ('RLS-DLA', '', {})
				, ('K-SVD', '', {})
				# , ('T-MOD', '', {})
				, ('K-HOSVD', '', {})
				]
			for title, transform_algorithm, kwargs in transform_algorithms:				
				t0 = time()
				dictionary = {}
				if title is 'MOD':
					dictMiniBatch = np.loadtxt(filePath + 'dictMODNoisy_' + fileNameSufix, delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeMODRefNoisy_' + fileNameSufix, delimiter=';')
				elif title is 'RLS-DLA':
					dictMiniBatch = np.loadtxt(filePath + 'dictRLS-DLANoisy_' + fileNameSufix, delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeRLS-DLARefNoisy_' + fileNameSufix, delimiter=';')
				elif title is 'K-SVD':
					dictMiniBatch = np.loadtxt(filePath + 'dictK-SVDNoisy_' + fileNameSufix, delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeK-SVDRefNoisy_' + fileNameSufix, delimiter=';')
				elif title is 'T-MOD':
					dictMiniBatch = np.loadtxt(filePath + 'dictT-MODNoisy_' + fileNameSufix, delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeT-MODRefNoisy_' + fileNameSufix, delimiter=';')
				elif title is 'K-HOSVD':
					dictMiniBatch = np.loadtxt(filePath + 'dictK-HOSVDNoisy_' + fileNameSufix, delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeK-HOSVDRefNoisy_' + fileNameSufix, delimiter=';')
				else:  # MiniBatchDL OMP
					t0 = time()
					miniBatch = MiniBatchDictionaryLearning(n_components=K, alpha=1, n_iter=noIt)
					miniBatchDict = miniBatch.fit(data.values).components_
					dictionary = miniBatchDict;
					dt = time() - t0
					# print >> results, 'miniBatchDict: ' + str(miniBatchDict.shape)
					# print >> results, 'Fit Time: %.2fs.' % dt
					print >> results, "\n" + title + ' L={:d}_N={:d}_tnz={:d}_K={:d}_noIt={:d}'.format(L, N, tnz, K, noIt)
					t0 = time()
					miniBatch.set_params(transform_algorithm=transform_algorithm, **kwargs)
					sparseCode = miniBatch.transform(data)
				reconstruction = np.dot(sparseCode, dictionary)
				reconstruction = pd.DataFrame(reconstruction, index=data.index.values)
				dt = time() - t0
				# print >> results, SparseCode: ' + str(sparseCode.shape)
				# print >> results, 'Reconstruction: ' + str(reconstruction.shape)
				# print >> results, 'Transform Time: %.2fs.' % dt
				print_comparison(reconstruction.values, data.values)
results.close()
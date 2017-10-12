# coding: utf-8
####################################################################################################################################################################
## Cross-Validation for evaluating the reconstruction performance by dictionary learning methods and parameter configurations applied to a fraud detection dataset
####################################################################################################################################################################

import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import matlab.engine
from time import time
from sklearn.decomposition import MiniBatchDictionaryLearning
from skimage.measure import (compare_mse, compare_nrmse, compare_psnr)

## Function to plot image difference
def print_comparison(image, reference, method_name):
	difference = image - reference
	mse = compare_mse(reference, image)
	nrmse = compare_nrmse(reference, image)
	psnr = compare_psnr(reference, image)
	text = method_name + ': norm: %(norm).4f\tMSE: %(MSE).4f\tNRMSE: %(NRMSE).4f\tPSNR: %(PSNR).4f' % {'norm': np.sqrt(np.sum(difference ** 2)), 'MSE': mse, 'NRMSE': nrmse, 'PSNR': psnr}
	print >> results, text

## settings
filePath = "/media/thiago/ubuntu/datasets/fraudDetection/train_under_data/"
results = open('results/fraud_detection_reconstruction_dl_cv.txt', 'w')
# matlab = matlab.engine.start_matlab()

## Load data
print >> results, '## Loading saved data...'
# data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/train_under_data.csv', index_col=0)
data = pd.read_csv('/media/thiago/ubuntu/datasets/fraudDetection/train_under_data.csv', index_col=0)
# print >> results, 'Data Shape: ' + str(data.shape)
# matlab.cd(r'/home/thiago/dev/projects/anomaly-detection/distributed-tensor-dictionary-learning', nargout=0)
# matlab.project(nargout=0)

## parameters
L = data.shape[0]
N = data.shape[1]
# K_range = [10, 50, 100, 200, 500]
# noIt_range = [10, 50, 100]
# tnz_range = [2, 3, 5, 7]
K_range = [10, 50, 100]
noIt_range = [10, 50, 100]
tnz_range = [2, 5]

## Compare to original one
np.set_printoptions(suppress=True)
print_comparison(data.values, data.values, 'Original Data')

## Load previously calculated data and evaluates their reconstruction performance for each parameter configuration
for K in K_range:
	for tnz in tnz_range:
		for noIt in noIt_range:
			print >> results, "\n" + title + ' L={:d}_N={:d}_tnz={:d}_K={:d}_noIt={:d}'.format(L, N, tnz, K, noIt)
			fileNameSufix = 'L={:d}_K={:d}_noIt={:d}_solver=javaORMP_tnz={:d}.csv'.format(L, K, noIt, tnz)
			transform_algorithms = [
				('MiniBatchDL OMP', 'omp', {'transform_n_nonzero_coefs': tnz})
				, ('MOD_javaORMP', '', {})
				, ('RLS-DLA_javaORMP', '', {})
				, ('K-SVD_javaORMP', '', {})
				# , ('T-MOD_javaORMP', '', {})
				# , ('K-HOSVD', '', {})
				]
			for title, transform_algorithm, kwargs in transform_algorithms:				
				dictionary = {}
				sparseCode = {}
				if title is 'MOD_javaORMP':
					dictionary = np.loadtxt(filePath + 'dictMODNoisy_' + fileNameSufix, delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeMODNoisy_' + fileNameSufix, delimiter=';')
				elif title is 'RLS-DLA_javaORMP':
					dictionary = np.loadtxt(filePath + 'dictRLS-DLANoisy_' + fileNameSufix, delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeRLS-DLANoisy_' + fileNameSufix, delimiter=';')
				elif title is 'K-SVD_javaORMP':
					dictionary = np.loadtxt(filePath + 'dictK-SVDNoisy_' + fileNameSufix, delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeK-SVDNoisy_' + fileNameSufix, delimiter=';')
				elif title is 'T-MOD_javaORMP':
					dictionary = np.loadtxt(filePath + 'dictT-MODNoisy_' + fileNameSufix, delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeT-MODNoisy_' + fileNameSufix, delimiter=';')
				elif title is 'K-HOSVD_javaORMP':
					dictionary = np.loadtxt(filePath + 'dictK-HOSVDNoisy_' + fileNameSufix, delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeK-HOSVDNoisy_' + fileNameSufix, delimiter=';')
				elif title is 'MiniBatchDL_OMP':
					miniBatch = MiniBatchDictionaryLearning(n_components=K, alpha=1, n_iter=noIt)
					dictionary = miniBatch.fit(data.values).components_
					miniBatch.set_params(transform_algorithm=transform_algorithm, **kwargs)
					sparseCode = miniBatch.transform(data)
				reconstruction = np.dot(sparseCode, dictionary)
				reconstruction = pd.DataFrame(reconstruction, index=data.index.values)
				np.set_printoptions(suppress=True)
				print(reconstruction.values)
				print_comparison(reconstruction.values, data.values)
results.close()

## ToDo: 
# Evaluate differents sparse coding solvers, such as OMP, Lasso, JavaOMP and others
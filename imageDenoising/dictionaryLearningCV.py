import numpy as np
# import pandas as pd
# import matlab.engine
from time import time
from sklearn.decomposition import MiniBatchDictionaryLearning
from skimage.measure import (compare_mse, compare_nrmse)


# Function to plot image difference
def print_comparison(target, reference, method_name):
	difference = target - reference
	mse = compare_mse(reference, target)
	nrmse = compare_nrmse(reference, target)
	psnr = 0
	text = method_name + ': norm: %(norm).4f\tMSE: %(MSE).4fs\tNRMSE: %(NRMSE).4fs\tPSNR: %(PSNR).4fs' % {'norm': np.sqrt(np.sum(difference ** 2)), 'MSE': mse, 'NRMSE': nrmse, 'PSNR': psnr}
	print >> results, text
	print(text)


# settings
filePath = "/media/thiago/ubuntu/datasets/imageDenoising/dl/"
results = open('results/imageDenoisingDLCV.txt', 'w')
# matlab = matlab.engine.start_matlab()


# Generating matlab data
# print('\n## Generating matlab data...')
# matlab.cd(r'/home/thiago/dev/projects/anomaly-detection/distributed-tensor-dictionary-learning', nargout=0)
# matlab.projectImageDenoising(nargout=0)


# print >> results, '\n## Loading saved data...'
# print('\n## Loading saved data...')
refPatches = np.loadtxt(filePath + 'refPatches.csv', delimiter=';')
noisyPatches = np.loadtxt(filePath + 'noisyPatches.csv', delimiter=';')
noiseMean = np.mean(noisyPatches, axis=0)
# print >> results, 'Data Shape: ' + str(refPatches.shape)
# print('Data Shape: ' + str(refPatches.shape))


patch_size = (7, 7)
L = refPatches.shape[0]
N = refPatches.shape[1]
K_range = [10, 50, 100]
noIt_range = [10, 50, 100]
tnz_range = [2, 5]


# ToDo: Evaluate differents sparse coding solvers, such as OMP, Lasso, JavaOMP and others
for K in K_range:
	for tnz in tnz_range:
		for noIt in noIt_range:

			fileNameSufix = 'L={:d}_K={:d}_noIt={:d}_solver=javaORMP_tnz={:d}.csv'.format(L, K, noIt, tnz)
			print >> results, '## L={:d}_N={:d}_tnz={:d}_K={:d}_noIt={:d}'.format(L, N, tnz, K, noIt)
			print('## L={:d}_N={:d}_tnz={:d}_K={:d}_noIt={:d}'.format(L, N, tnz, K, noIt))

			transform_algorithms = [
				('MiniBatchDL OMP', 'omp', {'transform_n_nonzero_coefs': tnz}),
				# ('RLS-DLA javaORMP', 'javaORMP', {'transform_n_nonzero_coefs': tnz}),
				# ('K-SVD javaORMP', 'javaORMP', {'transform_n_nonzero_coefs': tnz}),
				# ('T-MOD javaORMP', 'javaORMP', {'transform_n_nonzero_coefs': tnz}),
				# ('K-HOSVD javaORMP', 'javaORMP', {'transform_n_nonzero_coefs': tnz}),
				('MOD javaORMP', 'javaORMP', {'transform_n_nonzero_coefs': tnz})]

			for title, transform_algorithm, kwargs in transform_algorithms:
				t0 = time()
				dictionary = {}

				if title is 'MOD javaORMP':
					dictionary = np.loadtxt(filePath + 'dictMODNoisy_' + fileNameSufix, delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeMODNoisy_' + fileNameSufix, delimiter=';')
				elif title is 'RLS-DLA javaORMP':
					dictionary = np.loadtxt(filePath + 'dictRLS-DLANoisy_' + fileNameSufix, delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeRLS-DLANoisy_' + fileNameSufix, delimiter=';')
				elif title is 'K-SVD javaORMP':
					dictionary = np.loadtxt(filePath + 'dictK-SVDNoisy_' + fileNameSufix, delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeK-SVDNoisy_' + fileNameSufix, delimiter=';')
				elif title is 'T-MOD javaORMP':
					dictionary = np.loadtxt(filePath + 'dictT-MODNoisy_' + fileNameSufix, delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeT-MODNoisy_' + fileNameSufix, delimiter=';')
				elif title is 'K-HOSVD javaORMP':
					dictionary = np.loadtxt(filePath + 'dictK-HOSVDNoisy_' + fileNameSufix, delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeK-HOSVDNoisy_' + fileNameSufix, delimiter=';')
				elif title is 'MiniBatchDL OMP':
					t0 = time()
					miniBatch = MiniBatchDictionaryLearning(n_components=K, alpha=1, n_iter=noIt)
					dictionary = miniBatch.fit(refPatches).components_
					dt = time() - t0
					# print >> results, 'Fit Time: %.2fs.' % dt
					# print('Fit Time: %.2fs.' % dt)
					t0 = time()
					miniBatch.set_params(transform_algorithm=transform_algorithm, **kwargs)
					sparseCode = miniBatch.transform(noisyPatches)
				reconstruction = np.dot(sparseCode, dictionary)
				reconstruction += noiseMean
				# reconstruction = reconstruction.reshape(len(noisyPatches), *patch_size)
				dt = time() - t0
				# print('Reconstruction: ' + str(reconstruction.shape))
				# print >> results, 'Transform Time: %.2fs.' % dt
				# print('Transform Time: %.2fs.' % dt)
				print_comparison(reconstruction, refPatches)
results.close()

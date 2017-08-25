import numpy as np
# import pandas as pd
# import matlab.engine
# from time import time
from skimage.measure import (compare_mse, compare_nrmse, compare_psnr)
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d


# Function to plot image difference
def print_comparison(image, reference, method_name):
	difference = image - reference
	mse = compare_mse(reference, image)
	nrmse = compare_nrmse(reference, image)
	psnr = compare_psnr(reference, image)
	text = method_name + ': norm: %(norm).4f\tMSE: %(MSE).4f\tNRMSE: %(NRMSE).4f\tPSNR: %(PSNR).4f' % {'norm': np.sqrt(np.sum(difference ** 2)), 'MSE': mse, 'NRMSE': nrmse, 'PSNR': psnr}
	print >> results, text
	print(text)


# settings
filePath = "/media/thiago/ubuntu/datasets/imageDenoising/gray_192_512/"
results = open('results/print_gray_192_512_image_denoising_DL_CV.txt', 'w')
# matlab = matlab.engine.start_matlab()


# Generating matlab data
# print('\n## Generating matlab data...')
# matlab.cd(r'/home/thiago/dev/projects/anomaly-detection/distributed-tensor-dictionary-learning', nargout=0)
# matlab.projectImageDenoising(nargout=0)


# print >> results, '\n## Loading saved data...'
print('\n## Loading saved data...')
face = np.loadtxt(filePath + 'face.csv', delimiter=';')
distorted = np.loadtxt(filePath + 'distortedFace.csv', delimiter=';')
refPatches = np.loadtxt(filePath + 'refPatches.csv', delimiter=';')
# print >> results, 'Data Shape: ' + str(refPatches.shape)
# print('Data Shape: ' + str(refPatches.shape))


patch_size = (7, 7)
L = refPatches.shape[0]
N = refPatches.shape[1]
K_range = [10, 50, 100]
noIt_range = [10, 50, 100]
tnz_range = [2, 5]


print_comparison(distorted, face, 'Distorted image')


# Extract noisy patches and reconstruct them using the dictionary
print('\nExtracting noisy patches... ')
height, width = face.shape
noisyPatches = extract_patches_2d(distorted[:, width // 2:], patch_size)
noisyPatches = noisyPatches.reshape(noisyPatches.shape[0], -1)
noiseMean = np.mean(noisyPatches, axis=0)
noisyPatches -= noiseMean
# print('noisyPatches: ' + str(noisyPatches.shape))


# ToDo: Evaluate differents sparse coding solvers, such as OMP, Lasso, JavaOMP and others
for K in K_range:
	for tnz in tnz_range:
		for noIt in noIt_range:

			fileNameSufix = 'L={:d}_K={:d}_noIt={:d}_solver=javaORMP_tnz={:d}.csv'.format(L, K, noIt, tnz)
			print >> results, '\n## L={:d}_N={:d}_tnz={:d}_K={:d}_noIt={:d}'.format(L, N, tnz, K, noIt)
			print('\n## L={:d}_N={:d}_tnz={:d}_K={:d}_noIt={:d}'.format(L, N, tnz, K, noIt))

			reconstructions = {}
			transform_algorithms = [
				('MiniBatchDL OMP', 'omp', {'transform_n_nonzero_coefs': tnz}),
				('RLS-DLA javaORMP', 'javaORMP', {'transform_n_nonzero_coefs': tnz}),
				('K-SVD javaORMP', 'javaORMP', {'transform_n_nonzero_coefs': tnz}),
				# ('T-MOD javaORMP', 'javaORMP', {'transform_n_nonzero_coefs': tnz}),
				('K-HOSVD javaORMP', 'javaORMP', {'transform_n_nonzero_coefs': tnz}),
				('MOD javaORMP', 'javaORMP', {'transform_n_nonzero_coefs': tnz})]

			for title, transform_algorithm, kwargs in transform_algorithms:
				# t0 = time()
				reconstructions[title] = face.copy()
				dictionary = {}
				sparseCode = {}
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
					print('T-MOD dictionary: ' + str(dictionary.shape))
					print('T-MOD sparsecode: ' + str(sparseCode.shape))
				elif title is 'K-HOSVD javaORMP':
					dictionary = np.loadtxt(filePath + 'dictK-HOSVDNoisy_' + fileNameSufix, delimiter=';')
					sparseCode = np.loadtxt(filePath + 'sparseCodeK-HOSVDNoisy_' + fileNameSufix, delimiter=';')
				elif title is 'MiniBatchDL OMP':
					# t0 = time()
					miniBatch = MiniBatchDictionaryLearning(n_components=K, alpha=1, n_iter=noIt)
					dictionary = miniBatch.fit(noisyPatches).components_
					# dt = time() - t0
					# print >> results, 'Fit Time: %.2fs.' % dt
					# print('Fit Time: %.2fs.' % dt)
					# t0 = time()
					miniBatch.set_params(transform_algorithm=transform_algorithm, **kwargs)
					sparseCode = miniBatch.transform(noisyPatches)
				reconstruction = np.dot(sparseCode, dictionary)
				reconstruction += noiseMean
				reconstruction = reconstruction.reshape(len(noisyPatches), *patch_size)
				reconstructions[title][:, width // 2:] = reconstruct_from_patches_2d(reconstruction, (height, width // 2))
				# dt = time() - t0
				# print('Reconstruction: ' + str(reconstruction.shape))
				# print >> results, 'Transform Time: %.2fs.' % dt
				# print('Transform Time: %.2fs.' % dt)
				print_comparison(reconstructions[title], face, title)

results.close()

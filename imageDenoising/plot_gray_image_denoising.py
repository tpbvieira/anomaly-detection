"""
=========================================
Image denoising using dictionary learning
=========================================

An example comparing the effect of reconstructing noisy fragments of a raccoon face image using firstly online :ref:`DictionaryLearning` and various transform methods.

The dictionary is fitted on the distorted left half of the image, and subsequently used to reconstruct the right half. Note that even better performance could be achieved by fitting to an
undistorted (i.e. noiseless) image, but here we start from the assumption that it is not available.

A common practice for evaluating the results of image denoising is by looking at the difference between the reconstruction and the original image. If the reconstruction is perfect this will look
like Gaussian noise.

It can be seen from the plots that the results of :ref:`omp` with two non-zero coefficients is a bit less biased than when keeping only one (the edges look less prominent). It is in addition
closer from the ground truth in Frobenius norm.

The result of :ref:`least_angle_regression` is much more strongly biased: the difference is reminiscent of the local intensity value of the original image.
"""

import os.path
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from time import time
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version
from skimage.measure import (compare_mse, compare_nrmse, compare_psnr)


print(__doc__)


# Function to plot image difference
def plot_image_diff(image, reference, plot_title):
	"""Helper function to display denoising"""
	plt.figure(figsize=(5, 3.3))
	plt.subplot(1, 2, 1)
	plt.title('Image')
	plt.imshow(image, vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')
	plt.xticks(())
	plt.yticks(())
	plt.subplot(1, 2, 2)
	difference = image - reference
	mse = compare_mse(reference, image)
	nrmse = compare_nrmse(reference, image)
	psnr = compare_psnr(reference, image)
	subtitle = 'norm: %(norm).4f\nMSE: %(MSE).4fs\nNRMSE: %(NRMSE).4fs\nPSNR: %(PSNR).4fs' % {'norm': np.sqrt(np.sum(difference ** 2)), 'MSE': mse, 'NRMSE': nrmse, 'PSNR': psnr}
	print(plot_title + ': norm: %(norm).4f\tMSE: %(MSE).4fs\tNRMSE: %(NRMSE).4fs\tPSNR: %(PSNR).4fs' % {'norm': np.sqrt(np.sum(difference ** 2)), 'MSE': mse, 'NRMSE': nrmse, 'PSNR': psnr})
	plt.title(subtitle)
	plt.imshow(difference, vmin=-0.5, vmax=0.5, cmap=plt.cm.PuOr, interpolation='nearest')
	plt.xticks(())
	plt.yticks(())
	plt.suptitle(plot_title, size=16)
	plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)


# settings
patch_size = (7, 7)
filePath = "/media/thiago/ubuntu/datasets/imageDenoising/dl/"


# Load saved data or extract reference patches for dictionary learning and saves
if os.path.isfile(filePath + "face.csv") and os.path.isfile(filePath + "distortedFace.csv") and os.path.isfile(filePath + "refPatches.csv"):
	print('Loading saved data...')
	face = np.loadtxt(filePath + 'face.csv', delimiter=';')
	distorted = np.loadtxt(filePath + 'distortedFace.csv', delimiter=';')
	refPatches = np.loadtxt(filePath + 'refPatches.csv', delimiter=';')
	# print('Face: ' + str(face.shape))
	# print('Distorted: ' + str(distorted.shape))
	# print('RefPatches: ' + str(refPatches.shape))
else:
	if sp_version < (0, 12):
		raise SkipTest("Skipping because SciPy version earlier than 0.12.0 and thus does not include the scipy.misc.face() image.")

	# load face data set
	try:
		from scipy import misc
		face = misc.face(gray=True)
	except AttributeError:
		face = sp.face(gray=True) # Old versions of scipy have face in the top level package

	# test face picture
	# plt.imshow(face, cmap=plt.compare_msecm.gray)
	# plt.show()

	# Convert from uint8 representation, with values between 0 and 255, to a floating point representation, with values between 0 and 1.
	face = (face * 1.0) / 255

	# down sample for higher speed
	face = face[::2, ::2] + face[1::2, ::2] + face[::2, 1::2] + face[1::2, 1::2]
	face /= 4.0
	height, width = face.shape

	# Distort the right half of the image
	print('\nDistorting image...')
	distorted = face.copy()
	distorted[:, width // 2:] += 0.075 * np.random.randn(height, width // 2)

	# Extract all reference patches from the left half of the image
	print('\nExtracting reference patches...')
	t0 = time()
	refPatches = extract_patches_2d(distorted[:, :width // 2], patch_size)
	refPatches = refPatches.reshape(refPatches.shape[0], -1)
	refPatches -= np.mean(refPatches, axis=0)
	refPatches /= np.std(refPatches, axis=0)
	print('Extracting reference patches done in %.2fs.' % (time() - t0))

	# Save data
	np.savetxt(filePath + 'face.csv', face, fmt='%.6f', delimiter=';')
	np.savetxt(filePath + 'distortedFace.csv', distorted, fmt='%.6f', delimiter=';')
	np.savetxt(filePath + 'refPatches.csv', refPatches, fmt='%.6f', delimiter=';')
	print('Face: ' + str(face.shape))
	print('Distorted: ' + str(distorted.shape))
	print('RefPatches: ' + str(refPatches.shape))


# Plot dictionaries
# plt.figure(figsize=(4.2, 4))
# for i, comp in enumerate(dictMiniBatch[:100]):
# 	plt.subplot(10, 10, i + 1)
# 	plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r, interpolation='nearest')
# 	plt.xticks(())
# 	plt.yticks(())
# plt.suptitle('MiniBatch Dictionary learned from face patches\n' +  '%d patches' % (len(refPatches)), fontsize=16)
# plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)


# Extract noisy patches and reconstruct them using the dictionary
print('\nExtracting noisy patches... ')
t0 = time()
height, width = face.shape
noisyPatches = extract_patches_2d(distorted[:, width // 2:], patch_size)
noisyPatches = noisyPatches.reshape(noisyPatches.shape[0], -1)
noiseMean = np.mean(noisyPatches, axis=0)
noisyPatches -= noiseMean
np.savetxt(filePath + 'noisyPatches.csv', noisyPatches, fmt='%.6f', delimiter=';')
# print('noisyPatches: ' + str(noisyPatches.shape))
# print('Extracting noisy patches done in %.2fs.' % (time() - t0))


# Learning MiniBatch Dictionary
print('\nLearning MiniBatch Dictionary...\n')
t0 = time()
miniBatch = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500)
dictMiniBatch = miniBatch.fit(noisyPatches).components_
dt = time() - t0
np.savetxt(filePath + 'dictMiniBatch.csv', dictMiniBatch, fmt='%.6f', delimiter=';')
# print('Dictionary: ' + str(dictMiniBatch.shape))
# print('Learning MiniBatch Dictionary done in %.2fs.' % dt)


# Plot difference between the original and distorted face
# print('\nDistorted image')
plot_image_diff(distorted, face, 'Distorted image')


# compare dictionary learning methods for image reconstruction
reconstructions = {}
transform_algorithms = [
	('MiniBatch lars (Sparsity: 2)', 'lars', {'transform_n_nonzero_coefs': 2}),
	('MiniBatch lars (Sparsity: 5)', 'lars', {'transform_n_nonzero_coefs': 5}),
	('MiniBatch OMP (Sparsity: 2)', 'omp', {'transform_n_nonzero_coefs': 2}),
	('MiniBatch OMP (Sparsity: 5)', 'omp', {'transform_n_nonzero_coefs': 5}),
	('RLS-DLA javaORMP (Sparsity: 2)', '', {}),
	('RLS-DLA javaORMP (Sparsity: 5)', '', {}),
	# ('K-SVD javaORMP (Sparsity: 2)', '', {}),
	# ('T-MOD javaORMP (Sparsity: 2)', '', {}),
	# ('K-HOSVD javaORMP (Sparsity: 2)', '', {}),
	('MOD javaORMP (Sparsity: 2)', '', {}),
	('MOD javaORMP (Sparsity: 5)', '', {})]

for title, transform_algorithm, kwargs in transform_algorithms:
	# print("\n"+title + ':')
	reconstructions[title] = face.copy()
	t0 = time()
	if title is 'MOD javaORMP (Sparsity: 2)':
		dictMiniBatch = np.loadtxt(filePath + 'dictMODNoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=2.csv', delimiter=';')
		sparseCode = np.loadtxt(filePath + 'sparseCodeMODNoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=2.csv', delimiter=';')
	elif title is 'MOD javaORMP (Sparsity: 5)':
		dictMiniBatch = np.loadtxt(filePath + 'dictMODNoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=5.csv', delimiter=';')
		sparseCode = np.loadtxt(filePath + 'sparseCodeMODNoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=5.csv', delimiter=';')
	elif title is 'RLS-DLA javaORMP (Sparsity: 2)':
		dictMiniBatch = np.loadtxt(filePath + 'dictRLS-DLANoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=2.csv', delimiter=';')
		sparseCode = np.loadtxt(filePath + 'sparseCodeRLS-DLANoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=2.csv', delimiter=';')
	elif title is 'RLS-DLA javaORMP (Sparsity: 5)':
		dictMiniBatch = np.loadtxt(filePath + 'dictRLS-DLANoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=5.csv', delimiter=';')
		sparseCode = np.loadtxt(filePath + 'sparseCodeRLS-DLANoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=5.csv', delimiter=';')
	elif title is 'K-SVD javaORMP (Sparsity: 2)':
		dictMiniBatch = np.loadtxt(filePath + 'dictK-SVDNoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=2.csv', delimiter=';')
		sparseCode = np.loadtxt(filePath + 'sparseCodeK-SVDNoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=2.csv', delimiter=';')
	elif title is 'T-MOD javaORMP (Sparsity: 2)':
		dictMiniBatch = np.loadtxt(filePath + 'dictT-MODNoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=2.csv', delimiter=';')
		sparseCode = np.loadtxt(filePath + 'sparseCodeT-MODNoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=2.csv', delimiter=';')
	elif title is 'K-HOSVD javaORMP (Sparsity: 2)':
		dictMiniBatch = np.loadtxt(filePath + 'dictK-HOSVDNoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=2.csv', delimiter=';')
		sparseCode = np.loadtxt(filePath + 'sparseCodeK-HOSVDNoisy_L=94500_K=100_noIt=10_solver=javaORMP_tnz=2.csv', delimiter=';')
	else:
		miniBatch.set_params(transform_algorithm=transform_algorithm, **kwargs)
		sparseCode = miniBatch.transform(noisyPatches)
	recPatches = np.dot(sparseCode, dictMiniBatch)
	recPatches += noiseMean
	recPatches = recPatches.reshape(len(noisyPatches), *patch_size)
	# np.savetxt(title + 'RecPatches.csv', recPatches, fmt='%.6f', delimiter=';')
	reconstructions[title][:, width // 2:] = reconstruct_from_patches_2d(recPatches, (height, width // 2))
	dt = time() - t0
	# print('SparseCode: ' + str(sparseCode.shape))
	# print('RecPatches: ' + str(recPatches.shape))
	# print(title + ' done in %.2fs.' % dt)
	plot_image_diff(reconstructions[title], face, title)

plt.show()

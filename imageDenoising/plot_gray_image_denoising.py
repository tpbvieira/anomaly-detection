import os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from skimage.measure import (compare_mse, compare_nrmse, compare_psnr)
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d


# Function to plot image difference
def plot_image_diff(noisy, reference, plot_title):
	"""Helper function to display denoising"""
	difference = noisy - reference
	mse = compare_mse(reference, noisy)
	nrmse = compare_nrmse(reference, noisy)
	psnr = compare_psnr(reference, noisy)
	subtitle = 'norm: %(norm).4f\nMSE: %(MSE).4f\nNRMSE: %(NRMSE).4f\nPSNR: %(PSNR).4fdB' % {'norm': np.sqrt(np.sum(difference ** 2)), 'MSE': mse, 'NRMSE': nrmse, 'PSNR': psnr}
	print(plot_title + ': norm: %(norm).4f\tMSE: %(MSE).4f\tNRMSE: %(NRMSE).4f\tPSNR: %(PSNR).4fdB' % {'norm': np.sqrt(np.sum(difference ** 2)), 'MSE': mse, 'NRMSE': nrmse, 'PSNR': psnr})
	plt.gray()
	plt.subplot(1, 2, 1)
	plt.title('Noisy')
	plt.imshow(noisy)
	plt.xticks(())
	plt.yticks(())
	plt.subplot(1, 2, 2)
	plt.title(subtitle)
	plt.imshow(reference)
	plt.xticks(())
	plt.yticks(())


# settings
filePath = "/media/thiago/ubuntu/datasets/imageDenoising/gray_192_512/"
patch_size = (7, 7)


if os.path.isfile(filePath + "face.csv") and os.path.isfile(filePath + "distortedFace.csv") and os.path.isfile(filePath + "refPatches.csv"):
	print('## Loading saved data...')
	face = np.loadtxt(filePath + 'face.csv', delimiter=';')
	height, width = face.shape
	distorted = np.loadtxt(filePath + 'distortedFace.csv', delimiter=';')
	refPatches = np.loadtxt(filePath + 'refPatches.csv', delimiter=';')
	# print('##	Face: ' + str(face.shape))
	# print('##	Distorted: ' + str(distorted.shape))
	# print('##	RefPatches: ' + str(refPatches.shape))
	# plot_image_diff(distorted[:, width // 2:], face[:, width // 2:], 'noisy and reference')
	# plt.show()
else:
	# load face dataset
	face = misc.face(gray=True)
	print('##	Reference: ' + str(face.shape))
	# plt.imshow(face, cmap=plt.compare_mse.gray)
	# plt.show()

	# Convert from uint8 representation, with values between 0 and 255, to a floating point representation, with values between 0 and 1.
	face = (face * 1.0) / 255

	# down sample
	face = face[::2, ::2] + face[1::2, ::2] + face[::2, 1::2] + face[1::2, 1::2]
	face /= 4.0
	height, width = face.shape
	face = face[0:height//2, :]
	height, width = face.shape
	print('##	Under: ' + str(face.shape))
	# plt.imshow(face, cmap=plt.compare_mse.gray)
	# plt.show()

	# Distort the right half of the image
	distorted = face.copy()
	distorted[:, width // 2:] += 0.075 * np.random.randn(height, width // 2)
	print('##	Distorted: ' + str(distorted.shape))
	# plt.imshow(distorted, cmap=plt.compare_mse.gray)
	# plt.show()

	plot_image_diff(distorted[:, width // 2:], face[:, width // 2:], 'noisy and reference')
	# plt.show()

	# Extract all reference patches from the left half of the image
	refPatches = extract_patches_2d(distorted[:, :width // 2], patch_size)
	refPatches = refPatches.reshape(refPatches.shape[0], -1)
	refPatches -= np.mean(refPatches, axis=0)
	refPatches /= np.std(refPatches, axis=0)
	print('##	refPatches: ' + str(refPatches.shape))

	# Save data
	print('\n##	Saving data...')
	np.savetxt(filePath + 'face.csv', face, fmt='%.6f', delimiter=';')
	np.savetxt(filePath + 'distortedFace.csv', distorted, fmt='%.6f', delimiter=';')
	np.savetxt(filePath + 'refPatches.csv', refPatches, fmt='%.6f', delimiter=';')

# Extract noisy patches and reconstruct them using the dictionary
noisyPatches = extract_patches_2d(distorted[:, width // 2:], patch_size)
noisyPatches = noisyPatches.reshape(noisyPatches.shape[0], -1)
noiseMean = np.mean(noisyPatches, axis=0)
noisyPatches -= noiseMean
np.savetxt(filePath + 'noisyPatches.csv', noisyPatches, fmt='%.6f', delimiter=';')
# print('noisyPatches: ' + str(noisyPatches.shape))


# Plot difference between the original and distorted face
plot_image_diff(distorted, face, 'Distorted image')


# compare dictionary learning methods for image reconstruction
transform_algorithms = [
	# ('T-MOD javaORMP (Sparsity: 2)', '', {}),
	('K-HOSVD javaORMP (Sparsity: 5)', '', {}),
	# ('RLS-DLA javaORMP (Sparsity: 2)', '', {}),
	('RLS-DLA javaORMP (Sparsity: 5)', '', {}),
	# ('K-SVD javaORMP (Sparsity: 2)', '', {}),
	('K-SVD javaORMP (Sparsity: 5)', '', {}),
	# ('MOD javaORMP (Sparsity: 2)', '', {}),
	('MOD javaORMP (Sparsity: 5)', '', {}),
	# ('MiniBatch lars (Sparsity: 2)', 'lars', {'transform_n_nonzero_coefs': 2}),
	# ('MiniBatch lars (Sparsity: 5)', 'lars', {'transform_n_nonzero_coefs': 5}),
	# ('MiniBatch OMP (Sparsity: 2)', 'omp', {'transform_n_nonzero_coefs': 2}),
	('MiniBatch OMP (Sparsity: 5)', 'omp', {'transform_n_nonzero_coefs': 5})
]

reconstructions = {}
for title, transform_algorithm, kwargs in transform_algorithms:
	reconstructions[title] = face.copy()
	if title is 'MOD javaORMP (Sparsity: 2)':
		dictionary = np.loadtxt(filePath + 'dictMODNoisy_L=46500_K=50_noIt=50_solver=javaORMP_tnz=2.csv', delimiter=';')
		sparseCode = np.loadtxt(filePath + 'sparseCodeMODNoisy_L=46500_K=50_noIt=50_solver=javaORMP_tnz=2.csv', delimiter=';')
	elif title is 'MOD javaORMP (Sparsity: 5)':
		dictionary = np.loadtxt(filePath + 'dictMODNoisy_L=46500_K=100_noIt=50_solver=javaORMP_tnz=5.csv', delimiter=';')
		sparseCode = np.loadtxt(filePath + 'sparseCodeMODNoisy_L=46500_K=100_noIt=50_solver=javaORMP_tnz=5.csv', delimiter=';')
	elif title is 'RLS-DLA javaORMP (Sparsity: 2)':
		dictionary = np.loadtxt(filePath + 'dictRLS-DLANoisy_L=46500_K=50_noIt=50_solver=javaORMP_tnz=2.csv', delimiter=';')
		sparseCode = np.loadtxt(filePath + 'sparseCodeRLS-DLANoisy_L=46500_K=50_noIt=50_solver=javaORMP_tnz=2.csv', delimiter=';')
	elif title is 'RLS-DLA javaORMP (Sparsity: 5)':
		dictionary = np.loadtxt(filePath + 'dictRLS-DLANoisy_L=46500_K=100_noIt=50_solver=javaORMP_tnz=5.csv', delimiter=';')
		sparseCode = np.loadtxt(filePath + 'sparseCodeRLS-DLANoisy_L=46500_K=100_noIt=50_solver=javaORMP_tnz=5.csv', delimiter=';')
	elif title is 'K-SVD javaORMP (Sparsity: 2)':
		dictionary = np.loadtxt(filePath + 'dictK-SVDNoisy_L=46500_K=50_noIt=50_solver=javaORMP_tnz=2.csv', delimiter=';')
		sparseCode = np.loadtxt(filePath + 'sparseCodeK-SVDNoisy_L=46500_K=50_noIt=50_solver=javaORMP_tnz=2.csv', delimiter=';')
	elif title is 'K-SVD javaORMP (Sparsity: 5)':
		dictionary = np.loadtxt(filePath + 'dictK-SVDNoisy_L=46500_K=100_noIt=50_solver=javaORMP_tnz=5.csv', delimiter=';')
		sparseCode = np.loadtxt(filePath + 'sparseCodeK-SVDNoisy_L=46500_K=100_noIt=50_solver=javaORMP_tnz=5.csv', delimiter=';')
	elif title is 'T-MOD javaORMP (Sparsity: 5)':
		continue
	elif title is 'K-HOSVD javaORMP (Sparsity: 5)':
		dictionary = np.loadtxt(filePath + 'dictK-HOSVDNoisy_L=46500_K=100_noIt=50_solver=javaORMP_tnz=5.csv', delimiter=';')
		sparseCode = np.loadtxt(filePath + 'sparseCodeK-HOSVDNoisy_L=46500_K=100_noIt=50_solver=javaORMP_tnz=5.csv', delimiter=';')
	else:
		miniBatch = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=50)
		dictionary = miniBatch.fit(refPatches).components_
		miniBatch.set_params(transform_algorithm=transform_algorithm, **kwargs)
		sparseCode = miniBatch.transform(noisyPatches)
	recPatches = np.dot(sparseCode, dictionary)
	recPatches += noiseMean
	recPatches = recPatches.reshape(len(noisyPatches), *patch_size)

	if transform_algorithm == 'threshold':
		recPatches -= recPatches.min()
		recPatches /= recPatches.max()

	# Plot dictionaries
	# plt.figure(figsize=(4.2, 4))
	# for i, comp in enumerate(dictionary[:100]):
	# 	plt.subplot(10, 10, i + 1)
	# 	plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r, interpolation='nearest')
	# 	plt.xticks(())
	# 	plt.yticks(())
	# plt.suptitle('Dictionary learned from face patches\n' + '%d patches' % (len(refPatches)), fontsize=16)
	# plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
	# plt.show()

	reconstructions[title][:, width // 2:] = reconstruct_from_patches_2d(recPatches, (height, width // 2))
	plot_image_diff(reconstructions[title], face, title)

# plt.show()

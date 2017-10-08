import os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from skimage.measure import (compare_mse, compare_nrmse, compare_psnr)
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d, check_array, check_random_state, BaseEstimator


from itertools import product
import numbers
from scipy import sparse
from numpy.lib.stride_tricks import as_strided


## Function to plot image difference
def plot_image_diff(noisy, reference, plot_title):
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


def _compute_n_patches(i_h, i_w, p_h, p_w, max_patches=None):
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    all_patches = n_h * n_w

    if max_patches:
        if (isinstance(max_patches, (numbers.Integral))
                and max_patches < all_patches):
            return max_patches
        elif (isinstance(max_patches, (numbers.Real))
                and 0 < max_patches < 1):
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches


def my_extract_patches(arr, patch_shape=8, extraction_step=1):
    """Extracts patches of any n-dimensional array in place using strides.
    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted
    patch_shape : integer or tuple of length arr.ndim
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.
    extraction_step : integer or tuple of length arr.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
    """

    arr_ndim = arr.ndim
    
    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    print('##my_extract_patches arr_shape: ' + str(arr.shape))
    print('##my_extract_patches patch_shape: ' + str(patch_shape))
    print('##my_extract_patches extraction_step: ' + str(extraction_step))

    patch_strides = arr.strides

    slices = [slice(None, None, st) for st in extraction_step]
    print('##my_extract_patches slices: ' + str(slices))
    indexing_strides = arr[slices].strides
    
    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) // np.array(extraction_step)) + 1
    print('##my_extract_patches arr_shape: ' + str(np.array(arr.shape)))
    print('##my_extract_patches patch_shape: ' + str(np.array(patch_shape)))
    print('##my_extract_patches extraction_shape: ' + str(np.array(extraction_step)))
    print('##my_extract_patches patch_indices_shape: ' + str(patch_indices_shape))
  
    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    print('##my_extract_patches shape: ' + str(shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))
    
    patches = as_strided(arr, shape=shape, strides=strides)
    return patches


def my_extract_patches_2d(image, patch_size, max_patches=None, random_state=None):
    """Reshape a 2D image into a collection of patches
    The resulting patches are allocated in a dedicated array.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    image : array, shape = (image_height, image_width) or
        (image_height, image_width, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.
    patch_size : tuple of ints (patch_height, patch_width)
        the dimensions of one patch
    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.
    random_state : int, RandomState instance or None, optional (default=None)
        Pseudo number generator state used for random sampling to use if
        `max_patches` is not None.  If int, random_state is the seed used by
        the random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator is
        the RandomState instance used by `np.random`.
    Returns
    -------
    patches : array, shape = (n_patches, patch_height, patch_width) or
         (n_patches, patch_height, patch_width, n_channels)
         The collection of patches extracted from the image, where `n_patches`
         is either `max_patches` or the total number of patches that can be
         extracted.
    Examples
    --------
    >>> from sklearn.feature_extraction import image
    >>> one_image = np.arange(16).reshape((4, 4))
    >>> one_image
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> patches = image.extract_patches_2d(one_image, (2, 2))
    >>> print(patches.shape)
    (9, 2, 2)
    >>> patches[0]
    array([[0, 1],
           [4, 5]])
    >>> patches[1]
    array([[1, 2],
           [5, 6]])
    >>> patches[8]
    array([[10, 11],
           [14, 15]])
    """
    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    image = check_array(image, allow_nd=True)
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]

    extracted_patches = my_extract_patches(image, patch_shape=(p_h, p_w, n_colors), extraction_step=1)
    print('##my_extract_patches_2d extracted_patches: ' + str(extracted_patches.shape))
    print(extracted_patches * 255)

    n_patches = _compute_n_patches(i_h, i_w, p_h, p_w, max_patches)
    if max_patches:
        rng = check_random_state(random_state)
        i_s = rng.randint(i_h - p_h + 1, size=n_patches)
        j_s = rng.randint(i_w - p_w + 1, size=n_patches)
        patches = extracted_patches[i_s, j_s, 0]
    else:
        patches = extracted_patches

    patches = patches.reshape(-1, p_h, p_w, n_colors)
    print('##my_extract_patches_2d reshaped : ' + str(patches.shape))

    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        return patches.reshape((n_patches, p_h, p_w))
    else:
		return patches


## settings
patch_size = (2, 2)

## load face dataset
face = np.array([	[ 0,  1,  2,  3],
           			[ 4,  5,  6,  7],
           			[ 8,  9, 10, 11],
           			[12, 13, 14, 15]])
print('## image: ' + str(face.shape))
print(face)

## Converts from uint8 representation, with values between 0 and 255, to a floating point representation, with values between 0 and 1.
face = (face * 1.0) / 255
height, width = face.shape

## Extracts patches
patches = my_extract_patches_2d(face, patch_size)
print('## patches: ' + str(patches.shape))
np.set_printoptions(precision=4)
print(patches * 255)
patches = patches.reshape(patches.shape[0], -1)
print('## reshaped patches: ' + str(patches.shape))

## Recovers from patches
recPatches = patches
recPatches = recPatches.reshape(len(patches), *patch_size)
reconstruction = reconstruct_from_patches_2d(recPatches, (height, width))
plot_image_diff(reconstruction, face, 'reconstruction')
print(reconstruction * 255)
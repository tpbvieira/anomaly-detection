"""
====================
Denoising a picture
====================

In this example, we denoise a noisy version of a picture using the total
variation, bilateral, and wavelet denoising filters.

Total variation and bilateral algorithms typically produce "posterized" images
with flat domains separated by sharp edges. It is possible to change the degree
of posterization by controlling the tradeoff between denoising and faithfulness
to the original image.

Total variation filter
----------------------

The result of this filter is an image that has a minimal total variation norm,
while being as close to the initial image as possible. The total variation is
the L1 norm of the gradient of the image.

Bilateral filter
----------------

A bilateral filter is an edge-preserving and noise reducing filter. It averages
pixels based on their spatial closeness and radiometric similarity.

Wavelet denoising filter
------------------------

A wavelet denoising filter relies on the wavelet representation of the image.
The noise is represented by small values in the wavelet domain which are set to
0.

In color images, wavelet denoising is typically done in the `YCbCr color
space`_ as denoising in separate color channels may lead to more apparent
noise.

.. _`YCbCr color space`: https://en.wikipedia.org/wiki/YCbCr

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import (compare_mse, compare_nrmse, compare_psnr)
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float, color
from skimage.util import random_noise

#plt.imshow(face, cmap=plt.cm.gray)
#plt.show()

# cut the original image to img_original
img_original = img_as_float(data.chelsea()[100:250, 50:300])
#print(type(img_original))
#print(len(img_original))
#print(img_original.shape)

sigma = 0.155
img_noisy = random_noise(img_original, var=sigma**2)

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 5), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
plt.gray()

# Estimate the average noise standard deviation across color channels, via Robust wavelet-based estimator of the (Gaussian) noise standard deviation.
sigma_est_noise = estimate_sigma(img_noisy, multichannel=True, average_sigmas=True)

# Due to clipping in random_noise, the estimate will be a bit smaller than the specified sigma.
print("Estimated Gaussian noise standard deviation = {}".format(sigma_est_noise))

# Noisy
ax[0, 0].imshow(img_noisy)
ax[0, 0].axis('off')
mse = compare_mse(img_original, img_noisy)
nrmse = compare_nrmse(img_original, img_noisy)
psnr = compare_psnr(img_original, img_noisy)
title = 'Noisy' + '\nMSE: ' + str(mse)[:6] + '\nNRMSE: ' + str(nrmse)[:6] + '\nPSNR: ' + str(psnr)[:6]
ax[0, 0].set_title(title)

# TV 0.1
img_denoise_tv =  denoise_tv_chambolle(img_noisy, weight=0.1, multichannel=True)
ax[0, 1].imshow(img_denoise_tv)
ax[0, 1].axis('off')
mse = compare_mse(img_original, img_denoise_tv)
nrmse = compare_nrmse(img_original, img_denoise_tv)
psnr = compare_psnr(img_original, img_denoise_tv)
title = 'TV 0.1' + '\nMSE: ' + str(mse)[:6] + '\nNRMSE: ' + str(nrmse)[:6] + '\nPSNR: ' + str(psnr)[:6]
ax[0, 1].set_title(title)

# Bilateral 0.05
img_denoise_bilateral = denoise_bilateral(img_noisy, sigma_color=0.05, sigma_spatial=15, multichannel=True)
ax[0, 2].imshow(img_denoise_bilateral)
ax[0, 2].axis('off')
mse = compare_mse(img_original, img_denoise_bilateral)
nrmse = compare_nrmse(img_original, img_denoise_bilateral)
psnr = compare_psnr(img_original, img_denoise_bilateral)
title = 'Bilateral 0.05' + '\nMSE: ' + str(mse)[:6] + '\nNRMSE: ' + str(nrmse)[:6] + '\nPSNR: ' + str(psnr)[:6]
ax[0, 2].set_title(title)

# Wavelet denoising
img_denoise_wavelet = denoise_wavelet(img_noisy, multichannel=True)
ax[0, 3].imshow(img_denoise_wavelet)
ax[0, 3].axis('off')
mse = compare_mse(img_original, img_denoise_wavelet)
nrmse = compare_nrmse(img_original, img_denoise_wavelet)
psnr = compare_psnr(img_original, img_denoise_wavelet)
title = 'Wavelet denoising' + '\nMSE: ' + str(mse)[:6] + '\nNRMSE: ' + str(nrmse)[:6] + '\nPSNR: ' + str(psnr)[:6]
ax[0, 3].set_title(title)

# TV 0.2
img_denoise_tv = denoise_tv_chambolle(img_noisy, weight=0.2, multichannel=True)
ax[1, 1].imshow(img_denoise_tv)
ax[1, 1].axis('off')
mse = compare_mse(img_original, img_denoise_tv)
nrmse = compare_nrmse(img_original, img_denoise_tv)
psnr = compare_psnr(img_original, img_denoise_tv)
title = 'TV 0.2' + '\nMSE: ' + str(mse)[:6] + '\nNRMSE: ' + str(nrmse)[:6] + '\nPSNR: ' + str(psnr)[:6]
ax[1, 1].set_title(title)

# Bilateral 0.1
img_denoise_bilateral = denoise_bilateral(img_noisy, sigma_color=0.1, sigma_spatial=15, multichannel=True)
ax[1, 2].imshow(img_denoise_bilateral)
ax[1, 2].axis('off')
mse = compare_mse(img_original, img_denoise_bilateral)
nrmse = compare_nrmse(img_original, img_denoise_bilateral)
psnr = compare_psnr(img_original, img_denoise_bilateral)
title = 'Bilateral 0.1' + '\nMSE: ' + str(mse)[:6] + '\nNRMSE: ' + str(nrmse)[:6] + '\nPSNR: ' + str(psnr)[:6]
ax[1, 2].set_title(title)

# Wavelet denoising in YCbCr colorspace
img_denoise_wavelet = denoise_wavelet(img_noisy, multichannel=True, convert2ycbcr=True)
ax[1, 3].imshow(img_denoise_wavelet)
ax[1, 3].axis('off')
mse = compare_mse(img_original, img_denoise_wavelet)
nrmse = compare_nrmse(img_original, img_denoise_wavelet)
psnr = compare_psnr(img_original, img_denoise_wavelet)
title = 'Wavelet denoising\nin YCbCr colorspace' + '\nMSE: ' + str(mse)[:6] + '\nNRMSE: ' + str(nrmse)[:6] + '\nPSNR: ' + str(psnr)[:6]
ax[1, 3].set_title(title)

# Original
ax[1, 0].imshow(img_original)
ax[1, 0].axis('off')
mse = compare_mse(img_original, img_original)
nrmse = compare_nrmse(img_original, img_original)
psnr = compare_psnr(img_original, img_original)
title = 'Original' + '\nMSE: ' + str(mse)[:6] + '\nNRMSE: ' + str(nrmse)[:6] + '\nPSNR: ' + str(psnr)[:6]
ax[1, 0].set_title(title)

fig.tight_layout()

plt.show()

from matplotlib import pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from skimage.restoration import denoise_tv_bregman
from skimage.restoration import denoise_bilateral
from skimage.restoration import denoise_wavelet
from skimage.io import imread

img = imread('DataSet/ruido6.png')
dst1 = denoise_tv_chambolle(img, weight=0.2, multichannel=True)
dst2 = denoise_tv_bregman(img, weight=0.1, multichannel=True)
dst3 = denoise_bilateral(img, multichannel=True)
dst5 = denoise_wavelet(img, multichannel=True, rescale_sigma=False)

plt.figure(figsize=(9, 6))
plt.subplot(231), plt.imshow(img), plt.gray(), plt.title('Original')
plt.subplot(232), plt.imshow(dst1), plt.gray(), plt.title('TVchambolle')
plt.subplot(233), plt.imshow(dst2), plt.gray(), plt.title('TVbregman')
plt.subplot(234), plt.imshow(dst3), plt.gray(), plt.title('bilateral')
plt.subplot(235), plt.imshow(dst5), plt.gray(), plt.title('wavelet')
plt.tight_layout()
plt.show()
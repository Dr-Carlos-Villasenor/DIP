# Agregar Ruido a la imagen con OpenCV

from matplotlib import pyplot as plt
from skimage.util import random_noise
from skimage.io import imread

I = imread('DataSet/House.jpeg', as_gray = True)
gauss = random_noise(I, mode='gaussian', seed=None, clip=True)
gauss_lv = random_noise(I, mode='localvar', seed=None, clip=True)
poisson = random_noise(I, mode='poisson', seed=None, clip=True)
sp = random_noise(I, mode='s&p', seed=None, clip=True)

plt.figure(figsize=(12, 8))
plt.subplot(231), plt.imshow(I), plt.title('Original'), plt.gray()
plt.subplot(232), plt.imshow(gauss), plt.title('Gaussian'), plt.gray()
plt.subplot(233), plt.imshow(gauss_lv), plt.title('Gaussian localvar'), plt.gray()
plt.subplot(234), plt.imshow(poisson), plt.title('Poisson'), plt.gray()
plt.subplot(235), plt.imshow(sp), plt.title('Salt & Pepper'), plt.gray()
plt.show()
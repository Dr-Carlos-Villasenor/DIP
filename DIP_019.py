# Agregar Ruido a la imagen con OpenCV

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('DataSet/House.jpeg', cv2.IMREAD_GRAYSCALE)

# Gaussian noise
mu, sigma = 0.0, 20.0
noiseGause = np.random.normal(mu, sigma, img.shape)
imgGaussNoise = img + noiseGause
imgGaussNoise = np.uint8(cv2.normalize(imgGaussNoise, None, 0, 255, cv2.NORM_MINMAX))  # Normalized to [0255]

# Rayleigh noise
a = 30.0
noiseRayleigh = np.random.rayleigh(a, size=img.shape)
imgRayleighNoise = img + noiseRayleigh
imgRayleighNoise = np.uint8(cv2.normalize(imgRayleighNoise, None, 0, 255, cv2.NORM_MINMAX))  # Normalized to [0255]

# Gamma noise
a, b = 10.0, 2.5
noiseGamma = np.random.gamma(shape=b, scale=a, size=img.shape)
imgGammaNoise = img + noiseGamma
imgGammaNoise = np.uint8(cv2.normalize(imgGammaNoise, None, 0, 255, cv2.NORM_MINMAX))  # Normalized to [0255]

# Exponential noise
a = 50.0
noiseExponent = np.random.exponential(scale=a, size=img.shape)
imgExponentNoise = img + noiseExponent
imgExponentNoise = np.uint8(cv2.normalize(imgExponentNoise, None, 0, 255, cv2.NORM_MINMAX))  # Normalized to [0255]

# Uniform Noise
mean, sigma = 10, 100
a = 2 * mean - np.sqrt(12 * sigma)  # a = -14.64
b = 2 * mean + np.sqrt(12 * sigma)  # b = 54.64
noiseUniform = np.random.uniform(a, b, img.shape)
imgUniformNoise = img + noiseUniform
imgUniformNoise = np.uint8(cv2.normalize(imgUniformNoise, None, 0, 255, cv2.NORM_MINMAX))  # Normalized to [0255]

# SP noise
ps, pp = 0.05, 0.02
mask = np.random.choice((0, 0.5, 1), size=img.shape[:2], p=[pp, (1-ps-pp), ps])
imgChoiceNoise = img.copy()
imgChoiceNoise[mask==1] = 255
imgChoiceNoise[mask==0] = 0

plt.figure(figsize=(12, 8))

plt.subplot(231), plt.title("Gauss noise")
plt.imshow(imgGaussNoise), plt.gray()

plt.subplot(232), plt.title("Rayleigh noise")
plt.imshow(imgRayleighNoise), plt.gray()

plt.subplot(233), plt.title("Gamma noise")
plt.imshow(imgGammaNoise), plt.gray()

plt.subplot(234), plt.title("Exponential noise")
plt.imshow(imgExponentNoise), plt.gray()

plt.subplot(235), plt.title("Uniform noise")
plt.imshow(imgUniformNoise), plt.gray()

plt.subplot(236), plt.title("Salt-pepper noise")
plt.imshow(imgChoiceNoise), plt.gray()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))

plt.subplot(231), plt.title("Gauss noise")
histNP, bins = np.histogram(imgGaussNoise.flatten(), bins=255, range=[0, 255], density=True)
plt.bar(bins[:-1], histNP[:])

plt.subplot(232), plt.title("Rayleigh noise")
histNP, bins = np.histogram(imgRayleighNoise.flatten(), bins=255, range=[0, 255], density=True)
plt.bar(bins[:-1], histNP[:])

plt.subplot(233), plt.title("Gamma noise")
histNP, bins = np.histogram(imgGammaNoise.flatten(), bins=255, range=[0, 255], density=True)
plt.bar(bins[:-1], histNP[:])

plt.subplot(234), plt.title("Exponential noise")
histNP, bins = np.histogram(imgExponentNoise.flatten(), bins=255, range=[0, 255], density=True)
plt.bar(bins[:-1], histNP[:])

plt.subplot(235), plt.title("Uniform noise")
histNP, bins = np.histogram(imgUniformNoise.flatten(), bins=255, range=[0, 255], density=True)
plt.bar(bins[:-1], histNP[:])

plt.subplot(236), plt.title("Salt-pepper noise")
histNP, bins = np.histogram(imgChoiceNoise.flatten(), bins=255, range=[0, 255], density=True)
plt.bar(bins[:-1], histNP[:])

plt.tight_layout()
plt.show()
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('DataSet/House.jpeg', cv2.IMREAD_GRAYSCALE)

mu, sigma = 8.0, 30.0
imgGaussNoise = img + np.random.normal(mu, sigma, img.shape)
imgGaussNoise = np.uint8(cv2.normalize(imgGaussNoise, None, 0, 255, cv2.NORM_MINMAX))
dst = cv2.fastNlMeansDenoising(img, None, 10, 21, 7)

plt.figure(figsize=(18, 6))
plt.subplot(131), plt.imshow(img), plt.gray(), plt.title('Original')
plt.subplot(132), plt.imshow(imgGaussNoise), plt.gray(), plt.title('Con ruido')
plt.subplot(133), plt.imshow(dst), plt.gray(), plt.title('Sin ruido')
plt.tight_layout()
plt.show()
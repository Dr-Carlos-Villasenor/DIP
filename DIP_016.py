import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('DataSet/House.jpeg', cv2.IMREAD_GRAYSCALE)

kernel = np.zeros(img.shape)
cv2.rectangle(kernel, (0, 0), (10, 10), (255, 255, 255), -1)

# Fourier transform image
img_dft = np.fft.fft2(img)
img_dft = np.fft.fftshift(img_dft)

# Fourier transform kernel
k_dft = np.fft.fft2(kernel)
k_dft = np.fft.fftshift(k_dft)

new_dft = k_dft * img_dft

# Inverse Fourier Transform
new_dft = np.fft.ifftshift(new_dft)
new_image = np.fft.ifft2(new_dft)
new_image = np.abs(new_image)

# Draw pictures
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(img, 'gray')
ax1.title.set_text('Original image')

ax4 = fig.add_subplot(2, 3, 4)
ax4.imshow(np.log(np.abs(img_dft)))
ax4.title.set_text('Original image DFT')

ax2 = fig.add_subplot(2, 3, 2)
ax2.imshow(kernel, 'gray')
ax2.title.set_text('Kernel')

ax5 = fig.add_subplot(2, 3, 5)
ax5.imshow(np.log(np.abs(k_dft)))
ax5.title.set_text('Kernel DFT')

ax2 = fig.add_subplot(2, 3, 3)
ax2.imshow(new_image, 'gray')
ax2.title.set_text('Result')

ax5 = fig.add_subplot(2, 3, 6)
ax5.imshow(np.log(np.abs(new_dft)))
ax5.title.set_text('Result DFT')


plt.show()
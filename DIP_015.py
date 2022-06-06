import cv2
from matplotlib import pyplot as plt
import numpy as np

def highPassFiltering(img, size):
    h, w = img.shape[0:2]
    h1,w1 = int(h/2), int(w/2)
    img[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 0
    return img

def lowPassFiltering(img, size):
    h, w = img.shape[0:2]
    h1,w1 = int(h/2), int(w/2)
    img2 = np.zeros((h, w), np.uint8)
    img2[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 1
    img3 = img2 * img
    return img3

gray = cv2.imread('DataSet/House.jpeg', cv2.IMREAD_GRAYSCALE)

# Fourier transform
img_dft = np.fft.fft2(gray)
dft_shift = np.fft.fftshift(img_dft)

#High pass filter
dft_shift=lowPassFiltering(dft_shift, 350)
res = np.log(np.abs(dft_shift))


# Inverse Fourier Transform
idft_shift = np.fft.ifftshift(dft_shift)
ifimg = np.fft.ifft2(idft_shift)
ifimg = np.abs(ifimg)

# Draw pictures
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(gray, 'gray')
ax1.title.set_text('Original image')

ax1 = fig.add_subplot(1, 3, 2)
ax1.imshow(res)
ax1.title.set_text('DFT filter')

ax1 = fig.add_subplot(1, 3, 3)
ax1.imshow(np.int8(ifimg), cmap=plt.cm.gray)
ax1.title.set_text('Result')

plt.show()
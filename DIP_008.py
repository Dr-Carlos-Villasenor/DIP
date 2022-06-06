# Convolution from scratch
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read images
I = cv.imread('DataSet/Lenna.png', cv.IMREAD_GRAYSCALE)

# Create kernel
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

# Convolution!
N, M = I.shape
I_new = np.zeros((N-2, M-2))
for i in range(1, N-1):
    for j in range(1, M-1):
        I_new[i-1, j-1] = np.abs((I[i-1:i+2, j-1:j+2] * kernel).sum())

# Normalize output image
I_new = 255 * (I_new / I_new.max())
I_new = np.array(I_new, dtype='uint8')

# Draw images
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax1.axis("off")
ax1.title.set_text('Original')
ax2 = fig.add_subplot(1, 2, 2)
ax2.axis("off")
ax2.title.set_text("Filtrada")
ax1.imshow(I, cmap=plt.cm.gray)
ax2.imshow(I_new, cmap=plt.cm.gray)
plt.savefig('Prewitt.jpg')
plt.show()
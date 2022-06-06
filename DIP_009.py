# Convolution with OpenCV
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read images
I = cv.imread('DataSet/blur.png', cv.IMREAD_GRAYSCALE)

# Create kernel
kernel = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

I_new = cv.filter2D(I, -1, kernel)

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
plt.savefig('Filtrada.jpg')
plt.show()
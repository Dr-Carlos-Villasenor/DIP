# Convolution with OpenCV
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read images
I = cv.imread('DataSet/Lenna.png', cv.IMREAD_GRAYSCALE)

p = 0.3
I_noise = np.zeros(I.shape)
N, M = I.shape
for i in range(N):
    for j in range(M):
        r = np.random.rand()
        if p > r:
            I_noise[i, j] = 0
        elif (1-p) < r:
            I_noise[i, j] = 255
        else:
            I_noise[i, j] = I[i, j]


I_noise = np.float32(I_noise)
I_new = cv.medianBlur(I_noise, 5)

# Draw images
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 3, 1)
ax1.axis("off")
ax1.title.set_text('Original')
ax2 = fig.add_subplot(1, 3, 2)
ax2.axis("off")
ax2.title.set_text("Con ruido")
ax3 = fig.add_subplot(1, 3, 3)
ax3.axis("off")
ax3.title.set_text("Filtrada")
ax1.imshow(I, cmap=plt.cm.gray)
ax2.imshow(I_noise, cmap=plt.cm.gray)
ax3.imshow(I_new, cmap=plt.cm.gray)
plt.savefig('Mediana.jpg')
plt.show()
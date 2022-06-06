# Convolution with OpenCV
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read images
I = cv.imread('DataSet/Lenna.png', cv.IMREAD_GRAYSCALE)

# Create kernel

#kernel =(1/100)* np.ones((10,10))
#I_new = cv.blur(I, (10,10))
'''
kernel =(1/16)* np.array([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]])
'''

'''
kernel =        np.array([[-1, -1, -1],
                          [-1, 8, -1],
                          [-1, -1, -1]])
'''

'''
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
'''

'''
kernel = np.array([[-1, 0],
                   [0, 1]])
'''

#I_new = cv.GaussianBlur(I, (45,45), 0)
#I_new = cv.bilateralFilter(I, 9, 75, 75)

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
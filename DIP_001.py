# conda env list
# conda activate cv
# conda list
# cd Desktop
# cd CV
# python

import numpy as np
import cv2

img = np.zeros((3, 3), dtype=np.uint8)

img = cv2.imread('DataSet/Lenna.png')

type(img)
img.shape
img.size
img.dtype

cv2.imshow('Ventana', img)
cv2.waitKey()

img2 = cv2.imread('DataSet/Lenna.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('gray.jpg', img2)
cv2.imshow('Ventana', img2)
cv2.waitKey()

img_temp = np.zeros(img.shape, dtype=np.uint8)
img_temp[:,:,0] = img[:,:,0]
cv2.imshow('1', img_temp)
cv2.waitKey()

ret, bin = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('1', bin)
cv2.imwrite('bin.jpg', bin)
cv2.waitKey()


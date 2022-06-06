import numpy as np
import cv2 as cv

I = cv.imread('DataSet/Lenna.png', cv.IMREAD_GRAYSCALE)

I_n1 = np.zeros(I.shape, dtype=np.uint8)
n, m = I.shape
for i in range(n):
    for j in range(m):
        I_n1[i, j] = 255 - I[i, j]

cv.imshow('2', I_n1)

I_n2 = cv.bitwise_not(I)
cv.imshow('1', I_n2)

cv.imwrite('negative.jpg', I_n1)
cv.waitKey()
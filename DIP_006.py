import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('DataSet/test.png', cv.IMREAD_GRAYSCALE)
hist, bins = np.histogram(img.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.savefig('Original_image.jpg')
plt.show()

equ = cv.equalizeHist(img)
res = np.hstack((img, equ)) #stacking images side-by-side
cv.imwrite('hist_eq.jpg',res)

hist, bins = np.histogram(equ.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color='b')
plt.hist(equ.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.savefig('Original_image.jpg')
plt.show()
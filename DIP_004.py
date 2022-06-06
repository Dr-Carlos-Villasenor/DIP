import numpy as np
import cv2 as cv

I1 = cv.imread('DataSet/Lenna.png')

for gamma in [0.1, 0.5, 1.2, 2.2]:
    I2 = np.array(255 * np.power(I1/255.0, gamma), dtype='uint8')
    cv.imwrite('gamma_' + str(gamma) + '.jpg', I2)
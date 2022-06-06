import numpy as np
import cv2 as cv

img = cv.imread('DataSet/street.jpg')
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

edges = cv.Canny(gray_img, 300, 200)

cv.imshow('ventana', edges)
cv.waitKey()

minLineLength = 600
maxLineGap = 10
lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
print(lines.shape)
for i in range(lines.shape[0]):
    cv.line(img, (lines[i,0,0], lines[i,0,1]),
                 (lines[i,0,2], lines[i,0,3]),
                 (0, 0, 255), 2)

cv.imshow('ventana', img)
cv.waitKey()

cv.destroyAllWindows()
cv.imwrite('houghlines.jpg', img)

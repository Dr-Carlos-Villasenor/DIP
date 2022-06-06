import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('DataSet/Lenna.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
r_image, g_image, b_image = cv.split(img)

r_image_eq = cv.equalizeHist(r_image)
g_image_eq = cv.equalizeHist(g_image)
b_image_eq = cv.equalizeHist(b_image)

image_eq = cv.merge((r_image_eq, g_image_eq, b_image_eq))

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax1.axis("off")
ax1.title.set_text('Original')
ax2 = fig.add_subplot(1, 2, 2)
ax2.axis("off")
ax2.title.set_text("Equalized")

ax1.imshow(img)
ax2.imshow(image_eq)
plt.savefig('hist_equ_color.jpg')
plt.show()
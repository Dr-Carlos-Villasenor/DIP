import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('DataSet/A.png', 0)

kernel = np.ones((5, 5), np.uint8)

img_erosion = cv2.erode(img, kernel, iterations=1)
img_dilation = cv2.dilate(img, kernel, iterations=1)

plt.figure(figsize=(6, 3))
plt.subplot(131), plt.imshow(img), plt.gray(), plt.title('Original')
plt.subplot(132), plt.imshow(img_erosion), plt.gray(),  plt.title('Erosión')
plt.subplot(133), plt.imshow(img_dilation), plt.gray(),  plt.title('Dilatación')
plt.tight_layout()
plt.show()

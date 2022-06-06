import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('DataSet/ruido1.jpg', cv2.IMREAD_GRAYSCALE)
dst = cv2.fastNlMeansDenoising(img, None, 20, 11, 21)

plt.figure(figsize=(18, 6))
plt.subplot(121), plt.imshow(img), plt.gray(), plt.title('Original')
plt.subplot(122), plt.imshow(dst), plt.gray(), plt.title('Sin ruido')
plt.tight_layout()
plt.show()
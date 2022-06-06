import cv2
import numpy as np
import matplotlib.pyplot as plt

A =  cv2.imread('DataSet/A.png', 0)
A2 = cv2.imread('DataSet/A2.png', 0)
A3 = cv2.imread('DataSet/A3.png', 0)

kernel = np.ones((5, 5), np.uint8)

apertura = cv2.morphologyEx(A2, cv2.MORPH_OPEN, kernel)
cierre = cv2.morphologyEx(A3, cv2.MORPH_CLOSE, kernel)
gradiente = cv2.morphologyEx(A, cv2.MORPH_GRADIENT, kernel)

plt.figure(figsize=(6, 3))
plt.subplot(231), plt.imshow(A), plt.gray(), plt.title('A')
plt.subplot(232), plt.imshow(A2), plt.gray(),  plt.title('A2')
plt.subplot(233), plt.imshow(A3), plt.gray(),  plt.title('A3')
plt.subplot(234), plt.imshow(gradiente), plt.gray(), plt.title('Gradiente')
plt.subplot(235), plt.imshow(apertura), plt.gray(),  plt.title('Apertura')
plt.subplot(236), plt.imshow(cierre), plt.gray(),  plt.title('Cierre')
plt.tight_layout()
plt.show()

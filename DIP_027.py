import cv2
import matplotlib.pyplot as plt

im_gray = cv2.imread('DataSet/Lenna.png', cv2.IMREAD_GRAYSCALE)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
ret, thresh1 = cv2.threshold(im_gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(6, 3))
plt.subplot(121), plt.imshow(im_gray), plt.gray(), plt.title('Original')
plt.subplot(122), plt.imshow(thresh1),  plt.title('Umbralizada')
plt.tight_layout()
plt.show()
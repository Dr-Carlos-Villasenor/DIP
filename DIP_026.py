import cv2
import matplotlib.pyplot as plt

im_gray = cv2.imread('DataSet/pluto.jpg', cv2.IMREAD_GRAYSCALE)
im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)

plt.figure(figsize=(6, 3))
plt.subplot(121), plt.imshow(im_gray), plt.gray(), plt.title('Original')
plt.subplot(122), plt.imshow(im_color),  plt.title('Falso color')
plt.tight_layout()
plt.show()

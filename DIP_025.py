import cv2
from skimage import io
import matplotlib.pyplot as plt

#Needs 8 bit, not float.
color_opencv = cv2.imread('DataSet/Lenna.png',1)
gray_opencv = cv2.imread('DataSet/Lenna.png', 0)
color_opencv2 = cv2.cvtColor(color_opencv, cv2.COLOR_BGR2RGB)


color_skimage = io.imread('DataSet/Lenna.png', as_gray=False)
gray_skimage = io.imread('DataSet/Lenna.png', as_gray=True)

B, G, R = cv2.split(color_opencv)

plt.figure(figsize=(6, 6))
plt.subplot(221), plt.imshow(color_opencv2), plt.title('Original')
plt.subplot(222), plt.imshow(B), plt.gray(), plt.title('B')
plt.subplot(223), plt.imshow(G), plt.gray(), plt.title('G')
plt.subplot(224), plt.imshow(R), plt.gray(), plt.title('R')
plt.tight_layout()
plt.show()

##########################################################


hsv_image = cv2.cvtColor(color_skimage, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_image)

plt.figure(figsize=(6, 6))
plt.subplot(221), plt.imshow(color_opencv2), plt.title('Original')
plt.subplot(222), plt.imshow(h), plt.gray(), plt.title('H')
plt.subplot(223), plt.imshow(s), plt.gray(), plt.title('S')
plt.subplot(224), plt.imshow(v), plt.gray(), plt.title('V')
plt.tight_layout()
plt.show()

#####################################

lab_image = cv2.cvtColor(color_skimage, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(lab_image)

plt.figure(figsize=(6, 6))
plt.subplot(221), plt.imshow(color_opencv2), plt.title('Original')
plt.subplot(222), plt.imshow(Y), plt.gray(), plt.title('Y')
plt.subplot(223), plt.imshow(Cr), plt.gray(), plt.title('Cr')
plt.subplot(224), plt.imshow(Cb), plt.gray(), plt.title('Cb')

plt.tight_layout()
plt.show()


#####################################


lab_image = cv2.cvtColor(color_skimage, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab_image)

plt.figure(figsize=(6, 6))
plt.subplot(221), plt.imshow(color_opencv2), plt.title('Original')
plt.subplot(222), plt.imshow(L), plt.gray(), plt.title('L')
plt.subplot(223), plt.imshow(A), plt.gray(), plt.title('A')
plt.subplot(224), plt.imshow(B), plt.gray(), plt.title('B')
plt.tight_layout()
plt.show()

############################################

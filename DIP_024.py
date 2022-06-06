import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import inpaint
from skimage.io import imread

image_orig = imread('DataSet/MarcaDeAgua.jpg')
mask = imread('DataSet/MarcaDeAgua_mascara.jpg', as_gray=True)
mask = mask > 0.5

image_result = np.zeros(image_orig.shape)
image_result[:, :, 0] = inpaint.inpaint_biharmonic(image_orig[:, :, 0], mask)
image_result[:, :, 1] = inpaint.inpaint_biharmonic(image_orig[:, :, 1], mask)
image_result[:, :, 2] = inpaint.inpaint_biharmonic(image_orig[:, :, 2], mask)

plt.figure(figsize=(18, 6))
plt.subplot(131), plt.imshow(image_orig), plt.title('Original')
plt.subplot(132), plt.imshow(mask), plt.gray(), plt.title('Mascara')
plt.subplot(133), plt.imshow(image_result), plt.gray(), plt.title('Reconstruida')
plt.tight_layout()
plt.show()
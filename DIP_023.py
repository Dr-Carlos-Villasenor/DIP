import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.restoration import inpaint

image_orig = data.astronaut()

# Create mask with six block defect regions
mask = np.zeros(image_orig.shape[:-1], dtype=bool)
mask[20:60, 0:20] = 1
mask[160:180, 70:155] = 1
mask[30:60, 170:195] = 1
mask[-60:-30, 170:195] = 1
mask[-180:-160, 70:155] = 1
mask[-60:-20, 0:20] = 1

# add a few long, narrow defects
mask[200:205, -200:] = 1
mask[150:255, 20:23] = 1
mask[365:368, 60:130] = 1


image_defect = image_orig * ~mask[..., np.newaxis]

image_result = np.zeros(image_orig.shape)
image_result[:, :, 0] = inpaint.inpaint_biharmonic(image_defect[:, :, 0], mask)
image_result[:, :, 1] = inpaint.inpaint_biharmonic(image_defect[:, :, 1], mask)
image_result[:, :, 2] = inpaint.inpaint_biharmonic(image_defect[:, :, 2], mask)

plt.figure(figsize=(6, 6))
plt.subplot(221), plt.imshow(image_orig), plt.title('Original')
plt.subplot(222), plt.imshow(mask), plt.gray(), plt.title('Mascara')
plt.subplot(223), plt.imshow(image_defect), plt.gray(), plt.title('Con defectos')
plt.subplot(224), plt.imshow(image_result), plt.gray(), plt.title('Reconstruida')
plt.tight_layout()
plt.show()
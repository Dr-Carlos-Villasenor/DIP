import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

img = mpimg.imread('DataSet/paisaje.jpg')
img = np.array(img, dtype=np.float64) / 255

w, h, d = img.shape
n_classes = 5

img_array = np.reshape(img, (w * h, d))

image_array_sample = shuffle(img_array)[:1000]
kmeans = KMeans(n_clusters = n_classes).fit(image_array_sample)
labels = kmeans.predict(img_array)

img_labels = np.reshape(labels, (w,h))
img_out = np.zeros((w, h, d))
label_idx = 0
for i in range(w):
    for j in range(h):
        img_out[i][j][0] = kmeans.cluster_centers_[img_labels[i][j]][0]
        img_out[i][j][1] = kmeans.cluster_centers_[img_labels[i][j]][1]
        img_out[i][j][2] = kmeans.cluster_centers_[img_labels[i][j]][2]

print(kmeans.cluster_centers_)

plt.figure(figsize=(6, 3))
plt.subplot(121), plt.imshow(img), plt.gray(), plt.title('Original')
plt.subplot(122), plt.imshow(img_out),  plt.title('Imagen cuantizada')
plt.tight_layout()
plt.show()


import cv2 as cv
import matplotlib.pyplot as plt

# Read Images
img1 = cv.imread('DataSet/Estatua01.jpeg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('DataSet/Estatua02.jpeg', cv.IMREAD_GRAYSCALE)

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x:x.distance)


# Draw matches
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches, None)
plt.imshow(img3), plt.show()

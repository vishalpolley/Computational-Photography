"""Image transforms: Warp using affine transform"""

import cv2
import numpy as np

# Read image
img = cv2.imread('tech.png')
height, width = img.shape[:2]
cv2.imshow("Original", img)

# Affine transform (may include translation, rotation, scale)
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])  # original point locations
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])  # desired point locations after transform (from feature matching, e.g.)

M_aff = cv2.getAffineTransform(pts1, pts2)  # compute affine transform (needs exactly 3 corresponding point pairs)
print "Computed affine transform matrix:"
print M_aff

img_aff = cv2.warpAffine(img, M_aff, (width, height))  # warp original image by applying it
cv2.imshow("Warped by affine transform", img_aff)

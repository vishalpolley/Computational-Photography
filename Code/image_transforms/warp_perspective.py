"""Image transforms: Warp using perspective transform"""

import cv2
import numpy as np

# Read image
img = cv2.imread('berlin-wall-03.png')
height, width = img.shape[:2]
cv2.imshow("Original", img)

# Perspective transform
pts1 = np.float32([[220, 85], [223, 414], [602, 190], [616, 323]])
pts2 = np.float32([[75, 75], [75, 225], [500, 75], [500, 225]])

M_persp = cv2.getPerspectiveTransform(pts1, pts2)  # compute perspective transform (needs exactly 4 points)
print "Computed perspective transform matrix:"
print M_persp

img_persp = cv2.warpPerspective(img, M_persp, (600, 300))  # apply transform; last arg is output image size
cv2.imshow("Perspective transform", img_persp)

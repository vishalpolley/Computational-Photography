"""Image transforms: Rotation"""

import cv2
import numpy as np

# Read image
img = cv2.imread('tech.png')
height, width = img.shape[:2]
cv2.imshow("Original", img)

# Rotation around origin (0, 0)
M_rot = cv2.getRotationMatrix2D((0, 0), 45, 1)  # 45 deg, scale = 1
print "Rotation matrix (around origin):"
print M_rot
img_rot = cv2.warpAffine(img, M_rot, (width, height))
cv2.imshow("Rotation around origin", img_rot)

# Rotation around center (i.e. translation + rotation)
M_rot_center = cv2.getRotationMatrix2D((width / 2, height / 2), -45, 1)  # -45 deg, scale = 1
print "\nRotation matrix (around center):"
print M_rot_center
img_rot_center = cv2.warpAffine(img, M_rot_center, (width, height))
cv2.imshow("Rotation around center", img_rot_center)

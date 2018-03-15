"""Image transforms: Scale and shear"""

import cv2
import numpy as np

# Read image
img = cv2.imread('tech.png')
height, width = img.shape[:2]
cv2.imshow("Original", img)

# Scale (resize)
img_scaled = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)  # specify scaling factors
# OR:
#img_scaled = cv2.resize(img, (int(1.5 * width), int(1.5 * height)), interpolation=cv2.INTER_CUBIC)  # specify target size
cv2.imshow("Scale", img_scaled)

# Shear or skew (horizontal only)
M_shear = np.float32(
    [[1, 0.5, 0],
     [0,   1, 0]])
print "Shear matrix:"
print M_shear
img_shear = cv2.warpAffine(img, M_shear, (int(width + 0.5 * height), height))  # output image needs to be wider to accomodate stretched image
cv2.imshow("Shear", img_shear)

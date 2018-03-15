"""Image transforms: Translation"""

import cv2
import numpy as np

# Read image
img = cv2.imread('tech.png')
height, width = img.shape[:2]
cv2.imshow("Original", img)

# Translation
M_trans = np.float32(
    [[1, 0, 100],
     [0, 1,  50]])
print "Translation matrix:"
print M_trans
img_trans = cv2.warpAffine(img, M_trans, (width, height))  # note: last arg is size of output image (width, height)
cv2.imshow("Translation", img_trans)

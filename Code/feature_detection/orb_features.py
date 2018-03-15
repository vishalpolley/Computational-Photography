"""ORB Feature Detection

Based on: http://docs.opencv.org/trunk/doc/py_tutorials/py_feature2d/py_orb/py_orb.html
"""

import numpy as np
import cv2

# Read image
img = cv2.imread("three-musicians.png")
print "Image size: {}x{}".format(img.shape[1], img.shape[0])
cv2.imshow("Image", img)

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", img_gray)

# Initialize ORB detector object
orb = cv2.ORB()  # or cv2.SIFT() in OpenCV 2.4.9+

# Find keypoints and show them on original image (with scale and orientation)
kp = orb.detect(img_gray)  # optional arg: mask
print "{} keypoints found".format(len(kp))
img_kp = cv2.drawKeypoints(img, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", img_kp)

# Compute feature descriptors
kp, des = orb.compute(img_gray, kp)
print "Feature descriptors (128-element vector for each keypoint):"
print des

# Alt. one-step method
#kp, des = orb.detectAndCompute(img_gray)

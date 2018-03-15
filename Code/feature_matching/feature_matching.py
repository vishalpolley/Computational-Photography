"""Feature Detection and Matching

Based on: http://docs.opencv.org/trunk/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
"""

import numpy as np
import cv2

# Supplement missing drawMatches() function (only in OpenCV 3.0.0+)
def drawMatches(img1, kp1, img2, kp2, matches, flags):
    """Draw image features (keypoints) and lines joining matches.

    Source: http://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python#26227854

    img1, img2 - Grayscale images (may work with color images as well)
    kp1, kp2   - Detected list of keypoints through any of the OpenCV keypoint detection algorithms
    matches    - List of matches of corresponding keypoints through any OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype=np.uint8)

    # Place the first image to the left
    out[:rows1, :cols1, :] = np.dstack([img1, img1, img1]) if len(img1.shape) == 2 else img1

    # Place the next image to the right of it
    out[:rows2, cols1:(cols1 + cols2), :] = np.dstack([img2, img2, img2]) if len(img2.shape) == 2 else img2

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 5
        # colour red
        # thickness = -1 (filled)
        cv2.circle(out, (int(x1), int(y1)), 5, (0, 0, 255), -1)   
        cv2.circle(out, (int(x2) + cols1, int(y2)), 5, (0, 0, 255), -1)

        # Draw a line in between the two points
        # thickness = 2
        # colour green
        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (0, 255, 0), 2)

    # Return output image
    return out

cv2.drawMatches = drawMatches

# Read images
img1 = cv2.imread("box.png")  # objects: box.png, book.png, basmati.png
img2 = cv2.imread("scene.png")
print "Image 1 size: {}x{}".format(img1.shape[1], img1.shape[0])
print "Image 2 size: {}x{}".format(img2.shape[1], img2.shape[0])
cv2.imshow("Image 1", img1)
cv2.imshow("Image 2", img2)

# Convert to grayscale
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector object
orb = cv2.ORB()  # or cv2.SIFT() in OpenCV 2.4.9+

# Find keypoints, compute descriptors and show them on original image (with scale and orientation)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
print "Image 1: {} keypoints found".format(len(kp1))
print "Image 2: {} keypoints found".format(len(kp2))
img1_kp = cv2.drawKeypoints(img1, kp1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_kp = cv2.drawKeypoints(img2, kp2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Image 1: Keypoints", img1_kp)
cv2.imshow("Image 2: Keypoints", img2_kp)

# Create BFMatcher (Brute Force Matcher) object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)
print "{} matches found".format(len(matches))

# Sort them in the order of their distance
matches = sorted(matches, key = lambda x: x.distance)

# Draw first 10 matches
img_out = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], flags=2)
cv2.imshow("Matches", img_out)

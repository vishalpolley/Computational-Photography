"""Harris Corner Detection"""

import numpy as np
import cv2

# Read image
img = cv2.imread("octagon.png")
#print "Read image from file; size: {}x{}".format(img.shape[1], img.shape[0])  # [debug]
cv2.imshow("Image", img)

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", img_gray)

# Compute Harris corner detector response (params: block size, Sobel aperture, Harris alpha)
h_response = cv2.cornerHarris(img_gray, 2, 3, 0.04)
h_min, h_max, _, _ = cv2.minMaxLoc(h_response)  # for thresholding, display scaling
#print "Harris response: [min, max] = [{}, {}]".format(h_min, h_max)  # [debug]
cv2.imshow("Harris response", np.uint8((h_response - h_min) * (255.0 / (h_max - h_min))))

# Select corner pixels above threshold
h_thresh = 0.01 * h_max
_, h_selected = cv2.threshold(h_response, h_thresh, 1, cv2.THRESH_TOZERO)

# Pick corner pixels that are local maxima
img_out = img.copy()
img_out[h_selected > 0] = (0, 0, 255)  # mark pixels above threshold

nhood_size = 5  # neighborhood size for non-maximal suppression (odd)
nhood_r = int(nhood_size / 2)  # neighborhood radius = size / 2
corners = []  # list of corner locations as (x, y, response) tuples
for y in xrange(h_selected.shape[0]):
    for x in xrange(h_selected.shape[1]):
        if h_selected.item(y, x):
            h_value = h_selected.item(y, x)  # response value at (x, y)
            nhood = h_selected[(y - nhood_r):(y + nhood_r + 1), (x - nhood_r):(x + nhood_r + 1)]
            if not nhood.size:
                continue  # skip empty neighborhoods (which can happen at edges)
            local_max = np.amax(nhood)  # compute neighborhood maximum
            if h_value == local_max:
                corners.append((x, y, h_value))  # add to list of corners
                h_selected[(y - nhood_r):(y + nhood_r), (x - nhood_r):(x + nhood_r)] = 0  # suppress
                h_selected.itemset((y, x), h_value)  # retain maxima value to suppress others
                cv2.circle(img_out, (x, y), 5, (0, 255, 0))  # draw circle highlight (optional)

cv2.imshow("Suppressed Harris response", np.uint8(h_selected * (255.0 / h_max)))
cv2.imshow("Output", img_out)
print "\n".join("{} {}".format(corner[0], corner[1]) for corner in corners)

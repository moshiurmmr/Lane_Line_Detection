"""
This script detects lane lines in images of roads. This was developed as part of Udacity Self Driving Car
nano degree program.

Steps in detecting lane lines:
image -> gray image -> blur gray image -> mask the region of interest (ROI) -> Hough Transform -> draw lines
on blank image array
"""

# import necessary packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# read the image and convert it to grayscale
image = mpimg.imread('test_images/exit-ramp.jpg')
gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray_img)
# display the image in separate window
plt.show()

# apply Gaussian smoothing to image
kernel_size = 5
blur_gray_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)

# apply Canny
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray_img, low_threshold, high_threshold)
# check the Canny image
#plt.imshow(masked_edges)
#plt.show()

# create a masked edges image
mask = np.zeros_like(edges)
ignore_mask_color = 255 # white color vertices in the mask

# define four side of the polygon to mask
image_shape = image.shape
# vertices of the polygon
vertices = np.array([[(450, 300), (500, 300), (image_shape[1], image_shape[0]), (0, image_shape[0])]],
                    dtype=np.int)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)
#plt.imshow(mask)
#plt.show()


# apply Hough transform
# define Hough transform parameters
rho = 2 #
theta = np.pi/180
threshold = 15
min_line_length = 40
max_line_gap =20

# a blank image/array the same size as the test image
line_image = np.copy(image) * 0

# run Hough on Canny edge detected image
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
# check the Hough transform
#plt.imshow(lines)
#plt.show()

# draw the lines on the blank line_image
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
# check the blank image with lines added
plt.imshow(line_image)
plt.show()

# create a color binary image and combine with the line image
color_edges = np.dstack((edges, edges, edges)) # for three color channels

# draw the lines on the color edge image
color_line_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
#plt.imshow(color_line_edges)
#plt.show()

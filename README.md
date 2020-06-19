# Lane Line Detection
This project builds a computer vision (CV) application that detects lane lines from images and videos of roads.

This repository is inspired by the Finding Lane Lines project of Udacity Self-Driving Car Engineer Nanodegree program. The test images and videos used in this repository have been used from the program.

<--- This is a work in progress --->

The main steps used for detecting lane lines in an image are:
1. Color transform to gray 
2. Canny edge detection
3. Image filtering using Gaussian Blur
4. Detection of region of interest (ROI)
5. Detection of lines
6. Hough transform

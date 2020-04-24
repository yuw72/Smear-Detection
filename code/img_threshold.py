# Python program to illustrate 

import cv2 
import numpy as np 
import os
image1 = cv2.imread('C:/Users/YUW72/Desktop/Geospacial Visualization/Smear-Detection/results/gaussian_blur/gaussian_blur0.jpg') 
save_path = 'C:/Users/YUW72\Desktop/Geospacial Visualization/Smear-Detection/results/img_threshold'
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) 
filename0 = 'img_threshold00.jpg'
filename1 = 'img_threshold01.jpg'
thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
										cv2.THRESH_BINARY, 101, 5) 

thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
										cv2.THRESH_BINARY, 101, 5) 
cv2.imwrite(os.path.join(save_path, filename0), thresh1)
cv2.imwrite(os.path.join(save_path, filename1), thresh2)
# cv2.imshow('Adaptive Mean', thresh1) 
# cv2.imshow('Adaptive Gaussian', thresh2) 

if cv2.waitKey(0) & 0xff == 27: 
	cv2.destroyAllWindows() 

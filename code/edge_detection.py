import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
img = cv2.imread('C:/Users/YUW72/Desktop/Geospacial Visualization/Smear-Detection/results/img_threshold/img_threshold00.jpg',0)
save_path = save_path = 'C:/Users/YUW72\Desktop/Geospacial Visualization/Smear-Detection/results/edge_detection'
edges = cv2.Canny(img,100,200)
filename = 'edge_detection.jpg'
cv2.imwrite(os.path.join(save_path, filename), edges)
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()


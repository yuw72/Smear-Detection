import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('results/noisy_mask/test.png', 0)

kernel1 = np.ones((2,2),np.uint8)
kernel2 = np.ones((10,10),np.uint8)

result = cv2.erode(img,kernel1,iterations = 1)
result = cv2.dilate(result,kernel2,iterations = 1)

# plt.imshow(result, cmap='gray')
# plt.show()
plt.imsave('results/final_output/test.png', result, cmap='gray')
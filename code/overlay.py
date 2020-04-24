import cv2
import os
img1 = cv2.imread('results/edge_detection/edge_detection.jpg')
img2 = cv2.imread('data/sample_drive/cam_0/393408710.jpg')
save_path = 'results/final_output'
dst = cv2.addWeighted(img1,0.7,img2,0.3,0)
filename = "final_output0.jpg"
cv2.imwrite(os.path.join(save_path, filename), dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
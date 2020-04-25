import cv2
import os
#-----Reading the image-----------------------------------------------------
path_to_data = 'data/sample_drive/cam_0'
images = os.listdir(path_to_data)
save_path = 'results/contrast/100'
# image = "average_img_cam2.jpg"
i=0
for image in images:  
    print("image",image)
    if i>100:
        break
    image_path = os.path.join(path_to_data, image)
    img = cv2.imread(image_path, 1) 
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
 
    cv2.imwrite(os.path.join(save_path, f'gaussian_blur{i}.jpg'), final)
    i+=1
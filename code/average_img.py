import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def average_image(path, shape, save, N):

    images = os.listdir(path)
    aggregate = np.zeros(shape)
    filename = 'average_img_cam' + path[-1] + '.jpg'

    index = np.random.choice(len(images)-N, 1, replace=False)  # randomly sample N continuous photo from all images
    # print(index)

    for i in range(N):
        aggregate += cv2.imread(os.path.join(path, images[index[0]+i]), cv2.IMREAD_COLOR)

    average = aggregate / N
    cv2.imwrite(os.path.join(save, filename), average)
    print('saved!')

if __name__ == '__main__':

    img_shape = (2032, 2032, 3)  # shape of the image
    num = 200  # number of samples to take

    # pls change to your own directory
    path_to_data = 'D:/Northwestern University/2020Spring/Geospatial Visualization/HW/HW1/data/'
    save_path = 'D:/Northwestern University/2020Spring/Geospatial ' \
                'Visualization/HW/HW1/Smear-Detection/results/average_image'
    cams = os.listdir(path_to_data)

    for cam in cams:
        path_to_cam = os.path.join(path_to_data, cam)
        # print(path_to_cam)
        average_image(path_to_cam, img_shape, save_path, num)

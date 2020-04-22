import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def average_image(path, shape, save):
    images = os.listdir(path)
    aggregate = np.zeros(shape)
    filename = 'average_img_cam' + path[-1] + '.jpg'
    for image in images:
        aggregate += cv2.imread(os.path.join(path, image), cv2.IMREAD_COLOR)

    average = aggregate / len(images)
    cv2.imwrite(os.path.join(save, filename), average)


if __name__ == '__main__':

    img_shape = (2032, 2032, 3)

    # pls change to your own directory
    path_to_data = 'D:/Northwestern University/2020Spring/Geospatial Visualization/HW/HW1/data/'
    save_path = 'D:/Northwestern University/2020Spring/Geospatial ' \
                'Visualization/HW/HW1/Smear-Detection/results/average_image'
    cams = os.listdir(path_to_data)

    for cam in cams:
        path_to_cam = os.path.join(path_to_data, cam)
        # print(path_to_cam)
        average_image(path_to_cam, img_shape, save_path)

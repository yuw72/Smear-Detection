import cv2
import matplotlib.pyplot as plt
import os
import numpy as np


def grad(path, save, shape):
    images = os.listdir(path)
    aggregate = np.zeros(shape)
    filename = 'average_img_cam' + path[-1] + '.jpg'

    for image in images:
        # print(os.path.join(path, image))
        img = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)

        x_grad = cv2.Scharr(img, ddepth=cv2.CV_64F, dx=True, dy=False)
        y_grad = cv2.Scharr(img, ddepth=cv2.CV_64F, dx=False, dy=True)

        gradient = np.sqrt(x_grad**2+y_grad**2)
        aggregate += gradient

    average_grad = aggregate / len(images)

    cv2.imwrite(os.path.join(save, filename), average_grad)


if __name__ == '__main__':
    path_to_data = 'data/sample_drive'
    save_path = 'results/average_grad'
    img_shape = (2032, 2032)

    cams = os.listdir(path_to_data)

    for cam in cams:
        path_to_cam = os.path.join(path_to_data, cam)
        # print(path_to_cam)
        grad(path_to_cam, save_path, img_shape)


import os
import cv2
import numpy as np


def func(path, save, interval, shape):
    images = os.listdir(path)
    images.sort()
    avg = np.zeros(shape)
    grad = np.zeros(shape)
    filename = 'img_cam' + path[-1] + '.jpg'

    for index, image in enumerate(images):

        img = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
        img = cv2.GaussianBlur(img, (5, 5), 1)

        avg += img / interval

        x_grad = cv2.Scharr(img, ddepth=cv2.CV_64F, dx=True, dy=False)
        y_grad = cv2.Scharr(img, ddepth=cv2.CV_64F, dx=False, dy=True)

        gradient = np.sqrt(x_grad ** 2 + y_grad ** 2)
        grad += gradient / interval

        if index % interval == 0:
            cv2.imwrite(f'{save}/avg/{index}_{filename}', avg)
            cv2.imwrite(f'{save}/grad/{index}_{filename}', grad)
            avg = np.zeros(shape)
            grad = np.zeros(shape)


if __name__ == '__main__':
    path_to_data = 'data/sample_drive'
    save_path = 'results/res'

    img_shape = (2032, 2032)
    cams = os.listdir(path_to_data)

    for cam in cams:
        path_to_cam = os.path.join(path_to_data, cam)
        save_p = os.path.join(save_path, cam)

        func(path_to_cam, save_p, 100, img_shape)

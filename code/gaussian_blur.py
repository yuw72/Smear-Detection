import cv2
import matplotlib.pyplot as plt
import os


def gaussian_blur(path, k_size, sigma, save):
    images = os.listdir(path)
    i = 0
    for image in images:
        if image.endswith('.jpg'):
            img = cv2.imread(os.path.join(path, image), cv2.IMREAD_COLOR)

            img = cv2.GaussianBlur(img, k_size, sigma)
            cv2.imwrite(os.path.join(save, f'gaussian_blur{i}.jpg'), img)
            i += 1


if __name__ == '__main__':
    kernel_size = (5, 5)
    sigma = 1

    path_to_data = 'results/average_image'
    save_path = 'results/gaussian_blur'

    gaussian_blur(path_to_data, kernel_size, sigma, save_path)

import cv2
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys, getopt

def change_index(img, indices):
    size = len(img)
    new_indices = []
    values = []
    for i in indices:
        x = i//size
        y = i%size
        values.append(img[x,y])
        new_indices.append([x,y])
    return normalize(np.array(new_indices), img.shape[0]), np.array(values)

def normalize(arr, vmax):
    return 2*arr/vmax - 1  # to [-1,1]

def inormalize(arr, vmax):
    return (arr + 1) / 2 * vmax

def fit(values, indices, grid, img):
    iters = 10

    for i in range(iters):
        poly = PolynomialFeatures(3)
        features = poly.fit_transform(indices)
        reg = LinearRegression().fit(features, values)

        # test
        test_features = poly.fit_transform(grid)
        pred = reg.predict(test_features)
        pred = np.array(pred.reshape(img.shape[0], img.shape[0]))

        # Calculate difference between I and I0, and find inliners
        diff = np.abs(img - pred)
        ind = normalize(np.argwhere(diff < img*0.1), img.shape[0])

        # update indices and values
        indices = np.unique(np.concatenate((indices, ind)), axis=0)
        if len(indices) == img.shape[0] * img.shape[1]:
            break
        values = []
        for index in indices:
            values.append(img[int(inormalize(index[0], img.shape[0])), int(inormalize(index[1], img.shape[0]))])
        values = np.array(values).reshape([len(values), 1])
    
    return pred

if __name__ == '__main__':
    # default path
    img_path = 'results/average_image/test.png'
    grad_path = 'results/average_grad/test.png'
    threshold = 0.5

    # Handle system args
    try:
        opts, args = getopt.getopt(sys.argv[1:],"i:g:t:",["img=","gradient="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--img"):  # mode
            img_path = arg
        elif opt in ("-g", "--gradient"):
            grad_path = arg
        elif opt in ("-t", "--threshold"):
            threshold = float(arg)

    # Read grayscale image
    img = cv2.imread(img_path, 0)
    grad = cv2.imread(grad_path, 0)
    img = cv2.resize(img, (512, 512))
    grad = cv2.resize(grad, (512, 512))

    # Find the top 50% pixels
    tot_pixels = img.shape[0] * img.shape[1]
    ind_i = np.argpartition(img.flatten(), -tot_pixels//2)[-tot_pixels//2:]
    ind_i, values_i = change_index(img, ind_i)
    ind_g = np.argpartition(grad.flatten(), -tot_pixels//2)[-tot_pixels//2:]
    ind_g, values_g = change_index(grad, ind_g)

    # Fit the bivariate polynomial I0
    grid = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            grid.append([i,j])

    I0 = fit(values_i, ind_i, normalize(np.array(grid), img.shape[0]), img)
    I0_grad = fit(values_g, ind_g, normalize(np.array(grid), img.shape[0]), grad)

    # Calculate att and scattering
    a = grad / I0_grad
    b = img - I0 * a

    # Impose binary mask on smear
    bin_ind = np.argwhere(np.abs(a)<threshold)
    binary_mask = np.zeros((img.shape[0], img.shape[1]))
    for index in bin_ind:
        binary_mask[index[0], index[1]] = 1

    # plt.subplot(2,2,1)
    # plt.imshow(img, cmap='gray')
    # plt.subplot(2,2,2)
    # plt.imshow(grad, cmap='gray')
    # plt.subplot(2,2,3)
    # plt.imshow(a, cmap='gray')
    # plt.subplot(2,2,4)
    # cv2.imwrite('results/noisy_mask/cam0.png', binary_mask)
    plt.imsave('results/intermediate/a_test.png', a, cmap='gray')
    plt.imsave('results/intermediate/b_test.png', b, cmap='gray')
    plt.imsave('results/noisy_mask/test.png', binary_mask, cmap='gray')


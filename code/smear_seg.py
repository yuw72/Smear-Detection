import cv2
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def change_index(img, indices):
    size = len(img)
    new_indices = []
    values = []
    for i in indices:
        x = i//size
        y = i%size
        values.append(img[x,y])
        new_indices.append([x,y])
    return normalize(np.array(new_indices)), np.array(values)

def normalize(arr):
    return 2*arr/128 - 1

def inormalize(arr):
    return (arr + 1) / 2 * 128

def fit(values, indices, grid, img):
    iters = 10
    poly = PolynomialFeatures(3)
    test_features = poly.fit_transform(grid)

    for i in range(iters):
        features = poly.fit_transform(indices)
        reg = LinearRegression().fit(features, values)

        # test
        pred = reg.predict(test_features)
        pred = np.array(pred.reshape(img.shape[0], img.shape[0]))

        # Calculate difference between I and I0, and find inliners
        diff = np.abs(img - pred)
        ind = normalize(np.argwhere(diff < img*0.1))

        # update indices and values
        indices = np.unique(np.concatenate((indices, ind)), axis=0)
        if len(indices) == img.shape[0] * img.shape[1]:
            break
        values = []
        for index in indices:
            values.append(img[int(inormalize(index[0])), int(inormalize(index[1]))])
        values = np.array(values).reshape([len(values), 1])
    
    return pred


# Read grayscale image
img = cv2.imread('results/average_image/test.png', 0)  # 2032*2032
grad = cv2.imread('results/average_grad/test.png', 0)
img = cv2.resize(img, (128, 128))
grad = cv2.resize(grad, (128, 128))

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

I0 = fit(values_i, ind_i, normalize(np.array(grid)), img)
I0_grad = fit(values_g, ind_g, normalize(np.array(grid)), grad)

# Calculate att and scattering
a = grad / I0_grad
b = img - I0 * a

# Impose binary mask on smear
bin_ind = np.argwhere(a<-30)
binary_mask = np.ones((img.shape[0], img.shape[1]))
for index in bin_ind:
    binary_mask[index[0], index[1]] = 0

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.subplot(2,2,2)
plt.imshow(a, cmap='gray')
plt.subplot(2,2,3)
plt.imshow(b, cmap='gray')
plt.show()


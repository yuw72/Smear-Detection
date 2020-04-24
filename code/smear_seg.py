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
    return np.array(new_indices)/size, np.array(values)


def fit(values, indices, grid, img):
    iters = 100
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
        ind = np.argwhere(diff < img*0.1)/img.shape[0]
        
        # update indices and values
        indices = np.unique(np.concatenate((indices, ind)), axis=0)
        if len(indices) == img.shape[0] * img.shape[1]:
            break
        values = []
        for index in indices:
            values.append(img[int(index[0]*img.shape[0]), int(index[1]*img.shape[0])])
        values = np.array(values).reshape([len(values), 1])
    
    return pred


# Read grayscale image
img = cv2.imread('results/average_image/average_img_cam0.jpg', 0)  # 2032*2032
grad = cv2.imread('results/average_grad/average_img_cam0.jpg', 0)
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

I0 = fit(values_i, ind_i, np.array(grid)/img.shape[0], img)
I0_grad = fit(values_g, ind_g, np.array(grid)/grad.shape[0], grad)

a = grad / I0_grad
b = img - I0 * a

bin_ind = np.argwhere(a<-100)
binary_mask = np.ones((img.shape[0], img.shape[1]))
for index in bin_ind:
    binary_mask[index[0], index[1]] = 0

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.subplot(2,2,2)
plt.imshow(binary_mask, cmap='gray')
# plt.subplot(2,2,3)
# plt.imshow(b, cmap='gray', vmin=b_min, vmax=b_max)
plt.show()


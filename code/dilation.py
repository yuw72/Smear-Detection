import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys, getopt


if __name__ == '__main__':
    # default
    img_path = 'results/noisy_mask/test.png'
    k1 = 2
    k2 = 10

    # Handle system args
    try:
        opts, args = getopt.getopt(sys.argv[1:],"i:k1:k2:",["img=","kernel1=","kernel2="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--img"):  # mode
            img_path = arg
        elif opt in ("-k1", "--kernel1"):
            k1 = int(arg)
        elif opt in ("-k2", "--kernel2"):
            k2 = int(arg)

    img = cv2.imread(img_path, 0)

    kernel1 = np.ones((k1,k1),np.uint8)
    kernel2 = np.ones((k2,k2),np.uint8)

    result = cv2.erode(img,kernel1,iterations = 1)
    result = cv2.dilate(result,kernel2,iterations = 1)

    plt.imsave('results/final_output/test.png', result, cmap='gray')
    plt.imshow(result, cmap='gray')
    plt.show()
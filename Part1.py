import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function Declaration to add Gaussian noise
def add_gaussian_noise(image, mean=0, std_dev=25, seed=None):
    row, col = image.shape
    if seed is not None:
        np.random.seed(seed)  # Set random seed
    gauss = np.random.normal(mean, std_dev, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy

# Function Declaration to implement Otsu's thresholding algorithm
def otsu_thresholding(image):
    # Converting image 
    image = image.astype(np.uint8)
    # Computing histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # Normalizing histogram
    hist_norm = hist.ravel() / hist.sum()
    # Calculating probabilities
    prob = np.zeros(256)
    for i in range(256):
        prob[i] = i * hist_norm[i]
    # Initializing variables
    max_var = 0
    threshold = 0
    # Iterating through all possible thresholds
    for t in range(1, 256):
        q1 = prob[:t].sum()
        q2 = prob[t:].sum()
        if q1 == 0:
            m1 = 0
        else:
            m1 = (prob[:t] / q1).sum()
        if q2 == 0:
            m2 = 0
        else:
            m2 = (prob[t:] / q2).sum()
        var = q1 * q2 * ((m1 - m2) ** 2)
        if var > max_var:
            max_var = var
            threshold = t
    return threshold

# Function Declaration to plot the images
def plot_images(image, noisy_image, thresholded_image):
    plt.figure(figsize=(10, 6))
    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.subplot(132)
    plt.imshow(noisy_image, cmap='gray')
    plt.title('Noisy Image')
    plt.subplot(133)
    plt.imshow(thresholded_image, cmap='gray')
    plt.title('Thresholded Image')
    plt.show()


# Main function
if __name__ == "__main__":
    # Creating a test image
    image = np.zeros((100, 100), dtype=np.uint8)
    image[30:40, 30:40] = 255 # white object
    image[70:95, 70:95] = 100 # gray image
    # Adding Gaussian noise
    noisy_image = add_gaussian_noise(image, seed=42)  # Set a fixed seed
    # Applying Otsu's thresholding algorithm
    threshold = otsu_thresholding(noisy_image)
    # Thresholding the image
    _, thresholded_image = cv2.threshold(noisy_image, threshold, 255, cv2.THRESH_BINARY)
    # Displaying original image, noisy image, and thresholded image
    plot_images(image, noisy_image, thresholded_image)  
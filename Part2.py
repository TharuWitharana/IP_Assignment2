import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function Declaration for region growing segmentation
def region_growing(image, seeds, threshold):
    height, width = image.shape
    segmented = np.zeros_like(image)
    visited = np.zeros_like(image)
    stack = []
    # Converting image to float32
    image = image.astype(np.float32)
    for seed in seeds:
        stack.append(seed)
    while len(stack) > 0:
        current_point = stack.pop()
        y, x = current_point
        if visited[y, x] == 1:
            continue
        visited[y, x] = 1
        # Using float32 type 
        if abs(image[y, x] - image[seeds[0][0], seeds[0][1]]) <= threshold:
            segmented[y, x] = 255
        if y - 1 >= 0:
            stack.append((y - 1, x))
        if y + 1 < height:
            stack.append((y + 1, x))
        if x - 1 >= 0:
            stack.append((y, x - 1))
        if x + 1 < width:
            stack.append((y, x + 1))
    return segmented

# Function Declaration to plot the images
def display_images(color_image, segmented_image):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.subplot(122)
    plt.imshow(segmented_image, cmap='gray')
    plt.title('Segmented Image')
    plt.show()

# Main function
if __name__ == "__main__":
    color_image = cv2.imread('img1.jpg')
    # Converting the image to grayscale
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    # Defining seeds for region growing
    seeds = [(30, 30)]
    # Setting threshold for region growing
    threshold = 10
    # Performing region growing segmentation
    segmented_image = region_growing(gray_image, seeds, threshold)
    # Displaying original image and segmented image
    display_images(color_image, segmented_image) 

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt



class EdgeDetector:
    def __init__(self):
        pass

    # Convert to grayscale
    def grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Display grayscale image
    def show_gray_scale(self, image):
        plt.figure(figsize = (10,10))
        plt.imshow(image,cmap = 'gray')
        plt.axis('off')
        plt.show()

    # Apply Sobel filter
    def sobel_filter(self, image):
        # Add lines and columns to allow the convolution on frontiers
        image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

        # Sobel filter
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        return sobel_x, sobel_y, gradient_magnitude

     # Display gradient image
    def show_gradient(self, gradient):
        gradient_out = np.absolute(gradient)
        gradient_out = gradient_out * 255 / np.max(gradient_out)
        plt.figure(figsize=(10, 10))
        plt.imshow(gradient_out, cmap='binary')
        plt.axis('off')
        plt.show()

    # Display edges based on gradient strength
    def show_edges(self, gradient, threshold):
        edges = gradient > threshold
        plt.figure(figsize=(10, 10))
        plt.imshow(edges, cmap='binary')
        plt.axis('off')
        plt.show()


def main():
    # Load image
    image_path = os.path.join(os.getcwd(), '../TestImages', 'RGB.jpg')
    image = cv2.imread(image_path)

    # Create edge detector
    edge_detector = EdgeDetector()

    # Convert image to grayscale
    gray_image = edge_detector.grayscale(image)

    # Display grayscale image
    edge_detector.show_gray_scale(gray_image)

    # Apply Sobel filter
    sobel_x, sobel_y, gradient_magnitude = edge_detector.sobel_filter(gray_image)

    # Display gradient images
    edge_detector.show_gradient(sobel_x)
    edge_detector.show_gradient(sobel_y)
    edge_detector.show_gradient(gradient_magnitude)

     # Print gradient matrices
    print("Gx:", sobel_x)
    print("Gy:", sobel_y)

    # Display edges based on gradient strength
    edge_detector.show_edges(gradient_magnitude, 80)
    edge_detector.show_edges(gradient_magnitude, 150)


if __name__ == '__main__':
    main()
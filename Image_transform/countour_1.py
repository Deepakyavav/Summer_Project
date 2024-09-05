import cv2
import numpy as np
import os

def detect_and_draw_contours(image_path, save_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # Step 1: Threshold the image to get a binary image
    _, binary_image = cv2.threshold(image, 21, 255, cv2.THRESH_BINARY)

    # Step 2: Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total contours found: {len(contours)}")

    # Step 3: Draw all contours on a copy of the original image
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    # Step 4: Save the resulting image with contours drawn
    cv2.imwrite(save_path, contour_image)
    print(f"Contours drawn and saved to {save_path}")

# Example usage
image_path = '/mnt/DATA/Glucoma/Prob./RIGA/Subtracting_image/Differance_image/sample_image.png'
save_path = '/mnt/DATA/Glucoma/Prob./RIGA/Subtracting_image/Countours_detecting/contours_detected.png'
detect_and_draw_contours(image_path, save_path)

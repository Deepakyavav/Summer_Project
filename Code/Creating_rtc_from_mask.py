import cv2
import numpy as np
from scipy.spatial import distance
import pandas as pd
import os

def calculate_rim_thickness(fundus_img_path, cup_mask_path, disc_mask_path):
    # Load the images
    fundus_img = cv2.imread(fundus_img_path)
    cup_mask = cv2.imread(cup_mask_path, cv2.IMREAD_GRAYSCALE)
    disc_mask = cv2.imread(disc_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize masks to match the fundus image if needed
    cup_mask = cv2.resize(cup_mask, (fundus_img.shape[1], fundus_img.shape[0]))
    disc_mask = cv2.resize(disc_mask, (fundus_img.shape[1], fundus_img.shape[0]))
    
    # Find contours for the cup and disc masks
    contours_cup, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_disc, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming the largest contour is the one of interest
    cup_contour = max(contours_cup, key=cv2.contourArea)
    disc_contour = max(contours_disc, key=cv2.contourArea)
    
    # Initialize list to hold rim thickness values
    rim_thickness = []

    # Calculate rim thickness for 180 points (2-degree intervals)
    num_points = 180
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    disc_center = np.mean(disc_contour, axis=0).squeeze()

    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        # Compute the distances from the disc center in the given direction
        projected_points = disc_center + direction * np.max(distance.cdist(disc_center.reshape(1, -1), disc_contour.squeeze()))
        
        # Convert to correct shape
        projected_points = projected_points.reshape(1, -1)
        
        # Find the nearest point on the cup contour
        distances = distance.cdist(projected_points, cup_contour.squeeze())
        min_distance = np.min(distances)
        rim_thickness.append(min_distance)
    
    return rim_thickness

def is_image_file(filename):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    return filename.lower().endswith(valid_extensions)

def find_file_with_base_name(directory, base_name):
    for filename in os.listdir(directory):
        if filename.lower().startswith(base_name.lower()) and is_image_file(filename):
            return os.path.join(directory, filename)
    return None

def process_images(fundus_dir, cup_mask_dir, disc_mask_dir, output_csv):
    # Create a dictionary to hold the data
    data = {'Image_ID': []}
    num_points = 180

    # Add columns for each angle value
    for i in range(num_points):
        data[f'Angle_{i*2}'] = []

    # Get list of image files
    fundus_images = [f for f in os.listdir(fundus_dir) if is_image_file(f)]

    # Process each set of images
    for image_name in fundus_images:
        fundus_img_path = os.path.join(fundus_dir, image_name)
        
        # Get the file name without extension
        base_name = os.path.splitext(image_name)[0]
        
        # Look for corresponding mask files with any valid extension
        cup_mask_path = find_file_with_base_name(cup_mask_dir, base_name)
        disc_mask_path = find_file_with_base_name(disc_mask_dir, base_name)
        
        # Check if corresponding mask files exist
        if not (cup_mask_path and disc_mask_path):
            print(f"Skipping {image_name} - missing mask files")
            continue

        rim_thickness = calculate_rim_thickness(fundus_img_path, cup_mask_path, disc_mask_path)
        
        # Add the values to the dictionary
        data['Image_ID'].append(image_name)
        for j in range(num_points):
            data[f'Angle_{j*2}'].append(rim_thickness[j])

    # Convert the dictionary to a DataFrame and save as a CSV file
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

    print(f"Rim thickness values saved to {output_csv}")


# Example usage
fundus_dir = '/scratch/cs23m105/Imges_Multi_model/t1/1.0_Original_Fundus_Images'
cup_mask_dir = '/scratch/cs23m105/Imges_Multi_model/t1/cup'
disc_mask_dir = '/scratch/cs23m105/Imges_Multi_model/t1/disc'
output_csv = '/scratch/cs23m105/Imges_Multi_model/Code/rtc_file.csv'

process_images(fundus_dir, cup_mask_dir, disc_mask_dir, output_csv)
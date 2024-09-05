import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import pandas as pd
import logging
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Configuration
CONFIG = {
    'image_size': (256, 256),
    'batch_size': 16,
    'dataset_path': "/scratch/cs23m105/Imges_Multi_model/t1/1.0_Original_Fundus_Images",
    'disc_model_path': "/scratch/cs23m105/Imges_Multi_model/saved_models/best_disc_model.pth",
    'cup_model_path': "/scratch/cs23m105/Imges_Multi_model/saved_models/best_cup_model.pth",
    'output_csv_path': "/scratch/cs23m105/Imges_Multi_model/rtc_results.csv",
    'visualization_output_dir': "/scratch/cs23m105/Imges_Multi_model/visualization",
    'visualization_image_name': "example_image.jpg"  # Replace with an actual image name from your dataset
}

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_preprocessing_transforms():
    image_transforms = transforms.Compose([
        transforms.Resize(CONFIG['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return image_transforms

class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transforms = get_preprocessing_transforms()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transforms(image)
        return image_tensor, os.path.basename(img_path)

class UNet(nn.Module):
    def __init__(self, n_classes=2, in_channels=3):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.center = conv_block(512, 1024)
        self.dec4 = conv_block(1024 + 512, 512)
        self.dec3 = conv_block(512 + 256, 256)
        self.dec2 = conv_block(256 + 128, 128)
        self.dec1 = conv_block(128 + 64, 64)
        self.final = nn.Conv2d(64, n_classes, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        center = self.center(self.pool(e4))
        
        d4 = self.dec4(torch.cat([self.up(center), e4], 1))
        d3 = self.dec3(torch.cat([self.up(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], 1))
        
        return self.final(d1)
    
def load_model(model_path, device):
    model = UNet(n_classes=2, in_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def generate_mask(model, image, device):
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return mask

def calculate_rim_thickness(fundus_img, cup_mask, disc_mask):
    # Resize masks to match the fundus image if needed
    cup_mask = cv2.resize(cup_mask, (fundus_img.shape[1], fundus_img.shape[0]))
    disc_mask = cv2.resize(disc_mask, (fundus_img.shape[1], fundus_img.shape[0]))
    
    # Find contours for the cup and disc masks
    contours_cup, _ = cv2.findContours(cup_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_disc, _ = cv2.findContours(disc_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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

def visualize_and_save_transformations(image_tensor, filename, disc_model, cup_model, device):
    os.makedirs(CONFIG['visualization_output_dir'], exist_ok=True)
    
    # Original Image
    img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    img_np = img_np.astype(np.uint8)
    cv2.imwrite(os.path.join(CONFIG['visualization_output_dir'], "1_original.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    # Generate masks
    disc_mask = generate_mask(disc_model, image_tensor, device)
    cup_mask = generate_mask(cup_model, image_tensor, device)

    # Save masks
    cv2.imwrite(os.path.join(CONFIG['visualization_output_dir'], "2_disc_mask.png"), (disc_mask * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(CONFIG['visualization_output_dir'], "3_cup_mask.png"), (cup_mask * 255).astype(np.uint8))

    # Compute and plot RTC
    rtc = calculate_rim_thickness(img_np, cup_mask, disc_mask)
    plt.figure(figsize=(12, 6))
    plt.plot(range(180), rtc)
    plt.title(f"RTC Curve for {filename}")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("RTC")
    plt.savefig(os.path.join(CONFIG['visualization_output_dir'], "4_rtc_curve.png"))
    plt.close()

    # Overlay masks on original image
    overlay = img_np.copy()
    overlay[disc_mask == 1] = [255, 0, 0]  # Red for disc
    overlay[cup_mask == 1] = [0, 255, 0]   # Green for cup
    cv2.imwrite(os.path.join(CONFIG['visualization_output_dir'], "5_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    logging.info(f"Visualizations saved in {CONFIG['visualization_output_dir']}")

def process_images(device):
    dataset = ImageDataset(CONFIG['dataset_path'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    disc_model = load_model(CONFIG['disc_model_path'], device)
    cup_model = load_model(CONFIG['cup_model_path'], device)

    results = []
    visualization_done = False

    for batch_images, batch_filenames in dataloader:
        batch_images = batch_images.to(device)

        for image, filename in zip(batch_images, batch_filenames):
            if filename == CONFIG['visualization_image_name'] and not visualization_done:
                visualize_and_save_transformations(image, filename, disc_model, cup_model, device)
                visualization_done = True

            disc_mask = generate_mask(disc_model, image, device)
            cup_mask = generate_mask(cup_model, image, device)

            img_np = image.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            img_np = img_np.astype(np.uint8)

            rtc = calculate_rim_thickness(img_np, cup_mask, disc_mask)

            results.append({
                'image_name': filename,
                **{f'rtc_{i}': rtc[i] for i in range(180)}
            })

    df = pd.DataFrame(results)
    df.to_csv(CONFIG['output_csv_path'], index=False)
    logging.info(f"Results saved to {CONFIG['output_csv_path']}")


logging.info("Starting RTC Generation")
logging.info(f"Configuration: {CONFIG}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

start_time = time.time()
process_images(device)
total_time = time.time() - start_time
logging.info(f"Total processing time: {total_time:.2f} seconds")




# import os
# import time
# import datetime
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import cv2
# import pandas as pd
# import logging
# import matplotlib.pyplot as plt

# # Configuration
# CONFIG = {
#     'image_size': (256, 256),
#     'batch_size': 16,
#     'polar_height': 256,
#     'polar_width': 360,
#     'dataset_path': "/scratch/cs23m105/Imges_Multi_model/t1/1.0_Original_Fundus_Images",
#     'disc_model_path': "/scratch/cs23m105/Imges_Multi_model/saved_models/best_disc_model.pth",
#     'cup_model_path': "/scratch/cs23m105/Imges_Multi_model/saved_models/best_cup_model.pth",
#     'output_csv_path': "/scratch/cs23m105/Imges_Multi_model/rtc_results.csv",
#     'visualization_output_dir': "/scratch/cs23m105/Imges_Multi_model/visualization",
#     'visualization_image_name': "/scratch/cs23m105/Imges_Multi_model/t1/1.0_Original_Fundus_Images/3.png"  # Replace with an actual image name from your dataset
# }

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def get_preprocessing_transforms():
#     image_transforms = transforms.Compose([
#         transforms.Resize(CONFIG['image_size']),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     return image_transforms

# class ImageDataset(Dataset):
#     def __init__(self, image_dir):
#         self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#         self.transforms = get_preprocessing_transforms()

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert('RGB')
#         image_tensor = self.transforms(image)
#         return image_tensor, os.path.basename(img_path)

# class UNet(nn.Module):
#     def __init__(self, n_classes=2, in_channels=3):
#         super(UNet, self).__init__()

#         def conv_block(in_channels, out_channels):
#             return nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, 3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(out_channels, out_channels, 3, padding=1),
#                 nn.ReLU(inplace=True)
#             )

#         self.enc1 = conv_block(in_channels, 64)
#         self.enc2 = conv_block(64, 128)
#         self.enc3 = conv_block(128, 256)
#         self.enc4 = conv_block(256, 512)
#         self.center = conv_block(512, 1024)
#         self.dec4 = conv_block(1024 + 512, 512)
#         self.dec3 = conv_block(512 + 256, 256)
#         self.dec2 = conv_block(256 + 128, 128)
#         self.dec1 = conv_block(128 + 64, 64)
#         self.final = nn.Conv2d(64, n_classes, 1)
        
#         self.pool = nn.MaxPool2d(2)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#     def forward(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(self.pool(e1))
#         e3 = self.enc3(self.pool(e2))
#         e4 = self.enc4(self.pool(e3))
        
#         center = self.center(self.pool(e4))
        
#         d4 = self.dec4(torch.cat([self.up(center), e4], 1))
#         d3 = self.dec3(torch.cat([self.up(d4), e3], 1))
#         d2 = self.dec2(torch.cat([self.up(d3), e2], 1))
#         d1 = self.dec1(torch.cat([self.up(d2), e1], 1))
        
#         return self.final(d1)

# def load_model(model_path, device):
#     model = UNet(n_classes=2, in_channels=3).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#     return model

# def generate_mask(model, image, device):
#     with torch.no_grad():
#         output = model(image.unsqueeze(0).to(device))
#         mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
#     return mask

# def polar_to_cartesian(polar_img):
#     cart_img = cv2.warpPolar(polar_img, (CONFIG['image_size'][0], CONFIG['image_size'][1]), 
#                              (CONFIG['image_size'][0]//2, CONFIG['image_size'][1]//2), 
#                              min(CONFIG['image_size'])//2, cv2.WARP_INVERSE_MAP)
#     return cart_img

# def compute_rtc(disc_mask, cup_mask):
#     rtc = np.zeros(360)
#     center = (disc_mask.shape[0] // 2, disc_mask.shape[1] // 2)
#     for angle in range(360):
#         rad = np.deg2rad(angle)
#         x = int(center[0] + (CONFIG['image_size'][0] // 2) * np.cos(rad))
#         y = int(center[1] + (CONFIG['image_size'][1] // 2) * np.sin(rad))
#         line = np.zeros_like(disc_mask)
#         cv2.line(line, center, (x, y), 1, 1)
#         disc_points = np.where(disc_mask & line)
#         cup_points = np.where(cup_mask & line)
#         if len(disc_points[0]) > 0 and len(cup_points[0]) > 0:
#             disc_radius = np.sqrt((disc_points[0][-1] - center[0])**2 + (disc_points[1][-1] - center[1])**2)
#             cup_radius = np.sqrt((cup_points[0][-1] - center[0])**2 + (cup_points[1][-1] - center[1])**2)
#             rtc[angle] = disc_radius - cup_radius
#     return rtc

# def visualize_and_save_transformations(image_tensor, filename, disc_model, cup_model, device):
#     os.makedirs(CONFIG['visualization_output_dir'], exist_ok=True)
    
#     # Original Image
#     img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
#     img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
#     img_np = img_np.astype(np.uint8)
#     cv2.imwrite(os.path.join(CONFIG['visualization_output_dir'], "1_original.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

#     # Generate masks
#     disc_mask = generate_mask(disc_model, image_tensor, device)
#     cup_mask = generate_mask(cup_model, image_tensor, device)

#     # Save polar masks
#     cv2.imwrite(os.path.join(CONFIG['visualization_output_dir'], "2_disc_mask_polar.png"), (disc_mask * 255).astype(np.uint8))
#     cv2.imwrite(os.path.join(CONFIG['visualization_output_dir'], "3_cup_mask_polar.png"), (cup_mask * 255).astype(np.uint8))

#     # Convert to Cartesian and save
#     disc_mask_cart = polar_to_cartesian(disc_mask)
#     cup_mask_cart = polar_to_cartesian(cup_mask)
#     cv2.imwrite(os.path.join(CONFIG['visualization_output_dir'], "4_disc_mask_cartesian.png"), (disc_mask_cart * 255).astype(np.uint8))
#     cv2.imwrite(os.path.join(CONFIG['visualization_output_dir'], "5_cup_mask_cartesian.png"), (cup_mask_cart * 255).astype(np.uint8))

#     # Compute and plot RTC
#     rtc = compute_rtc(disc_mask_cart, cup_mask_cart)
#     plt.figure(figsize=(12, 6))
#     plt.plot(range(360), rtc)
#     plt.title(f"RTC Curve for {filename}")
#     plt.xlabel("Angle (degrees)")
#     plt.ylabel("RTC")
#     plt.savefig(os.path.join(CONFIG['visualization_output_dir'], "6_rtc_curve.png"))
#     plt.close()

#     # Overlay masks on original image
#     overlay = img_np.copy()
#     overlay[disc_mask_cart == 1] = [255, 0, 0]  # Red for disc
#     overlay[cup_mask_cart == 1] = [0, 255, 0]   # Green for cup
#     cv2.imwrite(os.path.join(CONFIG['visualization_output_dir'], "7_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

#     logging.info(f"Visualizations saved in {CONFIG['visualization_output_dir']}")

# def process_images(device):
#     dataset = ImageDataset(CONFIG['dataset_path'])
#     dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False)

#     disc_model = load_model(CONFIG['disc_model_path'], device)
#     cup_model = load_model(CONFIG['cup_model_path'], device)

#     results = []
#     visualization_done = False

#     for batch_images, batch_filenames in dataloader:
#         batch_images = batch_images.to(device)

#         for image, filename in zip(batch_images, batch_filenames):
#             if filename == CONFIG['visualization_image_name'] and not visualization_done:
#                 visualize_and_save_transformations(image, filename, disc_model, cup_model, device)
#                 visualization_done = True

#             disc_mask = generate_mask(disc_model, image, device)
#             cup_mask = generate_mask(cup_model, image, device)

#             disc_mask_cart = polar_to_cartesian(disc_mask)
#             cup_mask_cart = polar_to_cartesian(cup_mask)

#             rtc = compute_rtc(disc_mask_cart, cup_mask_cart)

#             results.append({
#                 'image_name': filename,
#                 **{f'rtc_{i}': rtc[i] for i in range(360)}
#             })

#     df = pd.DataFrame(results)
#     df.to_csv(CONFIG['output_csv_path'], index=False)
#     logging.info(f"Results saved to {CONFIG['output_csv_path']}")


# logging.info("Starting RTC Generation")
# logging.info(f"Configuration: {CONFIG}")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# logging.info(f"Using device: {device}")

# start_time = time.time()
# process_images(device)
# total_time = time.time() - start_time
# logging.info(f"Total processing time: {total_time:.2f} seconds")


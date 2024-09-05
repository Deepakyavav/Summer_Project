import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CONFIG = {
    'image_size': (256, 256),
    'dataset_path': "/scratch/cs23m105/Imges_Multi_model/t1/1.0_Original_Fundus_Images",
    'disc_model_path': "/scratch/cs23m105/Imges_Multi_model/saved_models/best_disc_model.pth",
    'cup_model_path': "/scratch/cs23m105/Imges_Multi_model/saved_models/best_cup_model.pth",
    'output_dir': "/scratch/cs23m105/Imges_Multi_model/generated_masks",
    'polar_height': 256,
    'polar_width': 360
}

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

def cartesian_to_polar(image, center, max_radius, angles):
    """Convert an image from Cartesian to polar coordinates."""
    x, y = np.indices((image.shape[0], image.shape[1]))
    
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    theta = np.arctan2(y - center[1], x - center[0])
    
    r_polar = np.linspace(0, max_radius, CONFIG['polar_height'])
    theta_polar = np.linspace(-np.pi, np.pi, angles)
    
    theta_polar, r_polar = np.meshgrid(theta_polar, r_polar)
    
    x_polar = r * np.cos(theta)
    y_polar = r * np.sin(theta)
    
    polar_image = cv2.remap(image, x_polar.astype(np.float32), y_polar.astype(np.float32), cv2.INTER_LINEAR)
    
    return polar_image

def polar_to_cartesian(polar_image, original_size):
    h, w = original_size
    center = (w // 2, h // 2)
    max_radius = min(center[0], center[1])
    
    y, x = np.ogrid[0:h, 0:w]
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    theta = np.arctan2(y - center[1], x - center[0])
    
    r_polar = np.linspace(0, max_radius, CONFIG['polar_height'])
    theta_polar = np.linspace(-np.pi, np.pi, CONFIG['polar_width'])
    
    r_polar, theta_polar = np.meshgrid(r_polar, theta_polar)
    
    x_polar = r * np.cos(theta)
    y_polar = r * np.sin(theta)
    
    cartesian_image = cv2.remap(polar_image, x_polar.astype(np.float32), y_polar.astype(np.float32), cv2.INTER_LINEAR)
    
    return cartesian_image

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image = image.resize(CONFIG['image_size'], Image.BILINEAR)
    image_np = np.array(image)
    
    center = (image_np.shape[1] // 2, image_np.shape[0] // 2)
    max_radius = min(center[0], center[1])
    
    image_polar = cartesian_to_polar(image_np, center, max_radius, CONFIG['polar_width'])
    image_polar = Image.fromarray(image_polar.astype('uint8'), 'RGB')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image_polar).unsqueeze(0), original_size

def generate_mask(model, image_tensor, original_size, device):
    with torch.no_grad():
        output = model(image_tensor.to(device))
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    mask_polar = (mask * 255).astype(np.uint8)
    mask_cartesian = polar_to_cartesian(mask_polar, original_size)
    
    return mask_cartesian

def save_mask(mask, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(mask).save(output_path)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load the pretrained models
    disc_model = UNet(n_classes=2, in_channels=3).to(device)
    cup_model = UNet(n_classes=2, in_channels=3).to(device)
    
    disc_model.load_state_dict(torch.load(CONFIG['disc_model_path'], map_location=device))
    cup_model.load_state_dict(torch.load(CONFIG['cup_model_path'], map_location=device))
    
    disc_model.eval()
    cup_model.eval()

    # Create output directories
    disc_output_dir = os.path.join(CONFIG['output_dir'], 'disc_masks')
    cup_output_dir = os.path.join(CONFIG['output_dir'], 'cup_masks')
    os.makedirs(disc_output_dir, exist_ok=True)
    os.makedirs(cup_output_dir, exist_ok=True)

    # Process images
    for image_name in os.listdir(CONFIG['dataset_path']):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(CONFIG['dataset_path'], image_name)
            logging.info(f"Processing image: {image_name}")
            
            image_tensor, original_size = preprocess_image(image_path)
            
            # Generate and save disc mask
            disc_mask = generate_mask(disc_model, image_tensor, original_size, device)
            save_mask(disc_mask, os.path.join(disc_output_dir, image_name))
            
            # Generate and save cup mask
            cup_mask = generate_mask(cup_model, image_tensor, original_size, device)
            save_mask(cup_mask, os.path.join(cup_output_dir, image_name))

    logging.info("Processing complete!")

if __name__ == "__main__":
    main()
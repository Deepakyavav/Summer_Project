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

# Configuration
CONFIG = {
    'image_size': (256, 256),
    'batch_size': 16,
    'polar_height': 256,
    'polar_width': 360,
    'dataset_path': "/scratch/cs23m105/Imges_Multi_model/t1/1.0_Original_Fundus_Images",
    'disc_model_path': "/scratch/cs23m105/Imges_Multi_model/saved_models/best_disc_model.pth",
    'cup_model_path': "/scratch/cs23m105/Imges_Multi_model/saved_models/best_cup_model.pth",
    'output_csv_path': "/scratch/cs23m105/Imges_Multi_model/rtc_results.csv"
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

def polar_to_cartesian(polar_img):
    cart_img = cv2.warpPolar(polar_img, (CONFIG['image_size'][0], CONFIG['image_size'][1]), 
                             (CONFIG['image_size'][0]//2, CONFIG['image_size'][1]//2), 
                             min(CONFIG['image_size'])//2, cv2.WARP_INVERSE_MAP)
    return cart_img

def compute_rtc(disc_mask, cup_mask):
    rtc = np.zeros(360)
    center = (disc_mask.shape[0] // 2, disc_mask.shape[1] // 2)
    for angle in range(360):
        rad = np.deg2rad(angle)
        x = int(center[0] + (CONFIG['image_size'][0] // 2) * np.cos(rad))
        y = int(center[1] + (CONFIG['image_size'][1] // 2) * np.sin(rad))
        line = np.zeros_like(disc_mask)
        cv2.line(line, center, (x, y), 1, 1)
        disc_points = np.where(disc_mask & line)
        cup_points = np.where(cup_mask & line)
        if len(disc_points[0]) > 0 and len(cup_points[0]) > 0:
            disc_radius = np.sqrt((disc_points[0][-1] - center[0])**2 + (disc_points[1][-1] - center[1])**2)
            cup_radius = np.sqrt((cup_points[0][-1] - center[0])**2 + (cup_points[1][-1] - center[1])**2)
            rtc[angle] = disc_radius - cup_radius
    return rtc

def process_images(device):
    dataset = ImageDataset(CONFIG['dataset_path'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    disc_model = load_model(CONFIG['disc_model_path'], device)
    cup_model = load_model(CONFIG['cup_model_path'], device)

    results = []

    for batch_images, batch_filenames in dataloader:
        batch_images = batch_images.to(device)

        for image, filename in zip(batch_images, batch_filenames):
            disc_mask = generate_mask(disc_model, image, device)
            cup_mask = generate_mask(cup_model, image, device)

            disc_mask_cart = polar_to_cartesian(disc_mask)
            cup_mask_cart = polar_to_cartesian(cup_mask)

            rtc = compute_rtc(disc_mask_cart, cup_mask_cart)

            results.append({
                'image_name': filename,
                **{f'rtc_{i}': rtc[i] for i in range(360)}
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


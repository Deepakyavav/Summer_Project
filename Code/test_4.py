import os
import time
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import accuracy_score, jaccard_score
import psutil
import GPUtil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration and hyperparameters
CONFIG = {
    'image_size': (256, 256),
    'batch_size': 64,
    'num_epochs': 100,
    'initial_learning_rate': 0.001,
    'lr_scheduler_step_size': 30,
    'lr_scheduler_gamma': 0.1,
    'dataset_path': "/scratch/cs23m105/Imges_Multi_model/t1/1.0_Original_Fundus_Images",
    'disc_masks_path': "/scratch/cs23m105/Imges_Multi_model/t1/disc",
    'cup_masks_path': "/scratch/cs23m105/Imges_Multi_model/t1/cup",
    'intermediate_output_dir': "/scratch/cs23m105/Imges_Multi_model/intermediate_outputs",
    'model_save_dir': "/scratch/cs23m105/Imges_Multi_model/saved_models"
}

def load_data(dir1, dir2):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    
    def base_name_without_extension(filename):
        base = os.path.basename(filename)
        return os.path.splitext(base)[0]
    
    dir1_images = {base_name_without_extension(f): os.path.join(dir1, f) for f in os.listdir(dir1) if os.path.splitext(f)[1].lower() in image_extensions}
    dir2_images = {base_name_without_extension(f): os.path.join(dir2, f) for f in os.listdir(dir2) if os.path.splitext(f)[1].lower() in image_extensions}
    
    image_paths = []
    mask_paths = []
    
    for name in dir1_images:
        if name in dir2_images:
            image_paths.append(dir1_images[name])
            mask_paths.append(dir2_images[name])
    
    return image_paths, mask_paths

def get_preprocessing_transforms():
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    
    return image_transforms, mask_transforms

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.image_transforms, self.mask_transforms = get_preprocessing_transforms()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            mask = Image.open(self.mask_paths[idx]).convert('L')

            img = img.resize(self.image_size, Image.BILINEAR)
            mask = mask.resize(self.image_size, Image.NEAREST)

            img = self.image_transforms(img)
            mask = self.mask_transforms(mask)

            mask = mask.squeeze(0).long()

            return img, mask
        except Exception as e:
            logging.error(f"Error loading image or mask at index {idx}: {str(e)}")
            return None

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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, save_path):
    best_val_loss = float('inf')
    train_losses, val_losses, train_dice_scores, val_dice_scores = [], [], [], []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss, train_dice = 0.0, 0.0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_dice += dice_coefficient(outputs, masks).item() * images.size(0)
            
            if i % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                val_dice += dice_coefficient(outputs, masks).item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        train_dice /= len(train_loader.dataset)
        val_dice /= len(val_loader.dataset)
        
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dice_scores.append(train_dice)
        val_dice_scores.append(val_dice)

        epoch_time = time.time() - epoch_start_time
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}, '
              f'Time: {epoch_time:.2f}s, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logging.info(f"Saved best model to {save_path}")

        # Save intermediate model
        intermediate_save_path = os.path.join(os.path.dirname(save_path), f"intermediate_model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), intermediate_save_path)
        logging.info(f"Saved intermediate model to {intermediate_save_path}")

        print_system_info()

    return train_losses, val_losses, train_dice_scores, val_dice_scores

def dice_coefficient(pred, target):
    smooth = 1e-5
    
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)
    
    pred = pred.float()
    target = target.float()
    
    if pred.dim() == 0:
        pred = pred.unsqueeze(0)
    
    if pred.dim() == 1:
        pred = (pred > 0.5).float()
    else:
        if pred.size(1) > 1:
            pred = torch.softmax(pred, dim=1)[:, 1]
        else:
            pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
    
    target = target.view(-1)
    pred = pred.view(-1)
    
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())
    
    accuracy = accuracy_score(all_targets, all_preds)
    jaccard = jaccard_score(all_targets, all_preds, average='weighted')
    
    dice = dice_coefficient(torch.tensor(all_preds), torch.tensor(all_targets)).item()
    
    return accuracy, jaccard, dice

def save_intermediate_images(images, masks, outputs, epoch, batch_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(min(5, images.size(0))):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        mask = masks[i].cpu().numpy()
        pred = torch.argmax(outputs[i], dim=0).cpu().numpy()
        
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(img)
        plt.title('Original Image')
        plt.subplot(132)
        plt.imshow(mask, cmap='gray')
        plt.title('Ground Truth')
        plt.subplot(133)
        plt.imshow(pred, cmap='gray')
        plt.title('Prediction')
        plt.savefig(os.path.join(output_dir, f'epoch_{epoch}_batch_{batch_idx}_sample_{i}.png'))
        plt.close()

def print_system_info():
    cpu_usage = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    ram_usage = ram.percent
    ram_used = ram.used / (1024 * 1024 * 1024)  # Convert to GB
    gpus = GPUtil.getGPUs()
    gpu_usage = gpus[0].memoryUsed if gpus else "N/A"
    logging.info(f"System Info - CPU: {cpu_usage}%, RAM: {ram_usage}% ({ram_used:.2f} GB used), GPU Memory: {gpu_usage}MB")

# def train_type1_model1(dataset_path, disc_masks_path):
#     logging.info("Starting training for Type 1 Model 1 (Disc Segmentation)")
#     start_time = time.time()

#     image_paths, mask_paths = load_data(dataset_path, disc_masks_path)
#     train_image_paths, temp_image_paths, train_mask_paths, temp_mask_paths = train_test_split(image_paths, mask_paths, test_size=0.3, random_state=42)
#     val_image_paths, test_image_paths, val_mask_paths, test_mask_paths = train_test_split(temp_image_paths, temp_mask_paths, test_size=0.5, random_state=42)

#     train_dataset = CustomDataset(train_image_paths, train_mask_paths, image_size=CONFIG['image_size'])
#     val_dataset = CustomDataset(val_image_paths, val_mask_paths, image_size=CONFIG['image_size'])
#     test_dataset = CustomDataset(test_image_paths, test_mask_paths, image_size=CONFIG['image_size'])

#     train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

#     model = UNet(n_classes=2, in_channels=3)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=CONFIG['initial_learning_rate'])
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG['lr_scheduler_step_size'], gamma=CONFIG['lr_scheduler_gamma'])

#     save_path = os.path.join(CONFIG['model_save_dir'], 'best_disc_model.pth')
#     train_losses,


# ... (previous code remains the same)

def train_type1_model1(dataset_path, disc_masks_path):
    logging.info("Starting training for Type 1 Model 1 (Disc Segmentation)")
    start_time = time.time()

    image_paths, mask_paths = load_data(dataset_path, disc_masks_path)
    train_image_paths, temp_image_paths, train_mask_paths, temp_mask_paths = train_test_split(image_paths, mask_paths, test_size=0.3, random_state=42)
    val_image_paths, test_image_paths, val_mask_paths, test_mask_paths = train_test_split(temp_image_paths, temp_mask_paths, test_size=0.5, random_state=42)

    train_dataset = CustomDataset(train_image_paths, train_mask_paths, image_size=CONFIG['image_size'])
    val_dataset = CustomDataset(val_image_paths, val_mask_paths, image_size=CONFIG['image_size'])
    test_dataset = CustomDataset(test_image_paths, test_mask_paths, image_size=CONFIG['image_size'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    model = UNet(n_classes=2, in_channels=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['initial_learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG['lr_scheduler_step_size'], gamma=CONFIG['lr_scheduler_gamma'])

    save_path = os.path.join(CONFIG['model_save_dir'], 'best_disc_model.pth')
    train_losses, val_losses, train_dice_scores, val_dice_scores = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, CONFIG['num_epochs'], save_path
    )
    
    total_time = time.time() - start_time
    logging.info(f"Total training time for Type 1 Model 1: {total_time:.2f} seconds")

    plot_curves(train_losses, val_losses, train_dice_scores, val_dice_scores, os.path.join(CONFIG['model_save_dir'], 'disc_model_curves.png'))

    model.load_state_dict(torch.load(save_path))
    accuracy, jaccard, dice = evaluate_model(model, test_loader, device)
    logging.info(f"Disc Model Test Results - Accuracy: {accuracy:.4f}, Jaccard: {jaccard:.4f}, Dice: {dice:.4f}")

    return model

def train_type1_model2(disc_model, dataset_path, cup_masks_path):
    logging.info("Starting training for Type 1 Model 2 (Cup Segmentation)")
    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_paths, mask_paths = load_data(dataset_path, cup_masks_path)
    train_image_paths, temp_image_paths, train_mask_paths, temp_mask_paths = train_test_split(image_paths, mask_paths, test_size=0.3, random_state=42)
    val_image_paths, test_image_paths, val_mask_paths, test_mask_paths = train_test_split(temp_image_paths, temp_mask_paths, test_size=0.5, random_state=42)

    def preprocess_image(image_path, disc_model, device):
        image = Image.open(image_path).convert('RGB')
        image = image.resize(CONFIG['image_size'], Image.BILINEAR)
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            disc_mask = torch.argmax(disc_model(image_tensor), dim=1).squeeze().cpu().numpy()
        
        kernel = np.ones((5, 5), np.uint8)
        dilated_disc_mask = cv2.dilate(disc_mask.astype(np.uint8), kernel, iterations=1)
        
        masked_image = cv2.bitwise_and(np.array(image), np.array(image), mask=dilated_disc_mask)
        return transforms.ToTensor()(masked_image)

    class PreprocessedDataset(Dataset):
        def __init__(self, image_paths, mask_paths, disc_model, device):
            self.image_paths = image_paths
            self.mask_paths = mask_paths
            self.disc_model = disc_model
            self.device = device

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img = preprocess_image(self.image_paths[idx], self.disc_model, self.device)
            mask = Image.open(self.mask_paths[idx]).convert('L')
            mask = mask.resize(CONFIG['image_size'], Image.NEAREST)
            mask = transforms.ToTensor()(mask).squeeze(0).long()
            return img, mask

    train_dataset = PreprocessedDataset(train_image_paths, train_mask_paths, disc_model, device)
    val_dataset = PreprocessedDataset(val_image_paths, val_mask_paths, disc_model, device)
    test_dataset = PreprocessedDataset(test_image_paths, test_mask_paths, disc_model, device)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    model = UNet(n_classes=2, in_channels=3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['initial_learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG['lr_scheduler_step_size'], gamma=CONFIG['lr_scheduler_gamma'])

    save_path = os.path.join(CONFIG['model_save_dir'], 'best_cup_model.pth')
    train_losses, val_losses, train_dice_scores, val_dice_scores = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, CONFIG['num_epochs'], save_path
    )
    
    total_time = time.time() - start_time
    logging.info(f"Total training time for Type 1 Model 2: {total_time:.2f} seconds")

    plot_curves(train_losses, val_losses, train_dice_scores, val_dice_scores, os.path.join(CONFIG['model_save_dir'], 'cup_model_curves.png'))

    model.load_state_dict(torch.load(save_path))
    accuracy, jaccard, dice = evaluate_model(model, test_loader, device)
    logging.info(f"Cup Model Test Results - Accuracy: {accuracy:.4f}, Jaccard: {jaccard:.4f}, Dice: {dice:.4f}")

    return model

def train_type2_model(disc_model, dataset_path, cup_masks_path):
    logging.info("Starting training for Type 2 Model (Cup Segmentation with Transfer Learning)")
    start_time = time.time()

    image_paths, mask_paths = load_data(dataset_path, cup_masks_path)
    train_image_paths, temp_image_paths, train_mask_paths, temp_mask_paths = train_test_split(image_paths, mask_paths, test_size=0.3, random_state=42)
    val_image_paths, test_image_paths, val_mask_paths, test_mask_paths = train_test_split(temp_image_paths, temp_mask_paths, test_size=0.5, random_state=42)

    train_dataset = CustomDataset(train_image_paths, train_mask_paths, image_size=CONFIG['image_size'])
    val_dataset = CustomDataset(val_image_paths, val_mask_paths, image_size=CONFIG['image_size'])
    test_dataset = CustomDataset(test_image_paths, test_mask_paths, image_size=CONFIG['image_size'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    cup_model = UNet(n_classes=2, in_channels=3)
    cup_model.load_state_dict(disc_model.state_dict())
    cup_model.final = nn.Conv2d(64, 2, 1)  # Modify the final layer for binary segmentation
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cup_model = cup_model.to(device)

    for param in cup_model.parameters():
        param.requires_grad = False
    for param in cup_model.final.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cup_model.final.parameters(), lr=CONFIG['initial_learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG['lr_scheduler_step_size'], gamma=CONFIG['lr_scheduler_gamma'])

    save_path = os.path.join(CONFIG['model_save_dir'], 'best_cup_model_type2.pth')
    train_losses, val_losses, train_dice_scores, val_dice_scores = train_model(
        cup_model, train_loader, val_loader, criterion, optimizer, scheduler, device, CONFIG['num_epochs'], save_path
    )
    
    total_time = time.time() - start_time
    logging.info(f"Total training time for Type 2 Model: {total_time:.2f} seconds")

    plot_curves(train_losses, val_losses, train_dice_scores, val_dice_scores, os.path.join(CONFIG['model_save_dir'], 'cup_model_type2_curves.png'))

    cup_model.load_state_dict(torch.load(save_path))
    accuracy, jaccard, dice = evaluate_model(cup_model, test_loader, device)
    logging.info(f"Cup Model (Type 2) Test Results - Accuracy: {accuracy:.4f}, Jaccard: {jaccard:.4f}, Dice: {dice:.4f}")

    return cup_model

def plot_curves(train_losses, val_losses, train_dice_scores, val_dice_scores, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_dice_scores, label='Train Dice Score')
    plt.plot(val_dice_scores, label='Validation Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.title('Training and Validation Dice Score')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # parser = argparse.ArgumentParser(description="Glaucoma Segmentation Training")
    # parser.add_argument('--train_type', type=str, choices=['all', 'disc', 'cup_type1', 'cup_type2'], default='all', help='Specify which model to train')
    # args = parser.parse_args()

    # os.makedirs(CONFIG['model_save_dir'], exist_ok=True)
    # os.makedirs(CONFIG['intermediate_output_dir'], exist_ok=True)

    # logging.info("Starting Glaucoma Segmentation Training")
    # logging.info(f"Configuration: {CONFIG}")
    # print_system_info()

    # if args.train_type in ['all', 'disc']:
    #     # Train Type 1 Model 1 (Disc Segmentation)
    #     logging.info("\nTraining Disc Segmentation Model")
    #     disc_model = train_type1_model1(CONFIG['dataset_path'], CONFIG['disc_masks_path'])
    # else:
    #     disc_model = UNet(n_classes=2, in_channels=3)
    #     disc_model.load_state_dict(torch.load(os.path.join(CONFIG['model_save_dir'], 'best_disc_model.pth')))

    # if args.train_type in ['all', 'cup_type1']:
    #     # Train Type 1 Model 2 (Cup Segmentation)
    #     logging.info("\nTraining Cup Segmentation Model (Type 1)")
    #     cup_model_type1 = train_type1_model2(disc_model, CONFIG['dataset_path'], CONFIG['cup_masks_path'])

    # if args.train_type in ['all', 'cup_type2']:
    #     # Train Type 2 Model (Cup Segmentation with Transfer Learning)
    #     logging.info("\nTraining Cup Segmentation Model (Type 2)")
    #     cup_model_type2 = train_type2_model(disc_model, CONFIG['dataset_path'], CONFIG['cup_masks_path'])

    # logging.info("\nTraining Complete")
    # print_system_info()
    os.makedirs(CONFIG['model_save_dir'], exist_ok=True)
    os.makedirs(CONFIG['intermediate_output_dir'], exist_ok=True)

    logging.info("Starting Glaucoma Segmentation Training")
    logging.info(f"Configuration: {CONFIG}")
    print_system_info()

    # Training Disc Segmentation Model
    logging.info("\nTraining Disc Segmentation Model")
    disc_model = train_type1_model1(CONFIG['dataset_path'], CONFIG['disc_masks_path'])
    
    # If not training Disc Model, load pre-trained Disc Model
    disc_model = UNet(n_classes=2, in_channels=3)
    disc_model.load_state_dict(torch.load(os.path.join(CONFIG['model_save_dir'], 'best_disc_model.pth')))

    # Training Cup Segmentation Model (Type 1)
    logging.info("\nTraining Cup Segmentation Model (Type 1)")
    cup_model_type1 = train_type1_model2(disc_model, CONFIG['dataset_path'], CONFIG['cup_masks_path'])

    # Comment out if not needed
    #Training Cup Segmentation Model (Type 2)
    logging.info("\nTraining Cup Segmentation Model (Type 2)")
    cup_model_type2 = train_type2_model(disc_model, CONFIG['dataset_path'], CONFIG['cup_masks_path'])

    logging.info("\nTraining Complete")
    print_system_info()

if __name__ == "__main__":
    main()
import os
import time
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

# Hyperparameters and Configuration
CONFIG = {
    'image_size': (256, 256),
    'batch_size': 32,
    'num_epochs': 100,
    'initial_learning_rate': 0.001,
    'lr_scheduler_step_size': 30,
    'lr_scheduler_gamma': 0.1,
    'dataset_path': "/scratch/cs23m105/Imges_Multi_model/t1/1.0_Original_Fundus_Images",
    'disc_masks_path': "/scratch/cs23m105/Imges_Multi_model/t1/disc",
    'cup_masks_path': "/scratch/cs23m105/Imges_Multi_model/t1/cup",
    'intermediate_output_dir': "/scratch/cs23m105/Imges_Multi_model/intermediate_outputs"
}

def get_preprocessing_transforms():
    """Define preprocessing transforms for both images and masks."""
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
        img = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        img = img.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        img = self.image_transforms(img)
        mask = self.mask_transforms(mask)

        mask = mask.squeeze(0).long()

        return img, mask

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

def dice_coefficient(pred, target):
    smooth = 1e-5
    pred = torch.softmax(pred, dim=1)[:, 1]  # Get probabilities for positive class
    pred = (pred > 0.5).float()  # Convert to binary
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, save_path):
    best_val_loss = float('inf')
    train_losses, val_losses, train_dice_scores, val_dice_scores = [], [], [], []

    for epoch in range(num_epochs):
        start_time = time.time()
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
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

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

        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}, '
              f'Time: {epoch_time:.2f}s, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        # Print system information
        print_system_info()

    return train_losses, val_losses, train_dice_scores, val_dice_scores

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
    dice = np.mean([dice_coefficient(torch.tensor(p).unsqueeze(0), torch.tensor(t).unsqueeze(0)).item() 
                    for p, t in zip(all_preds, all_targets)])
    
    return accuracy, jaccard, dice

def save_intermediate_images(images, masks, outputs, epoch, batch_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(min(5, images.size(0))):  # Save up to 5 images
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
    ram_usage = psutil.virtual_memory().percent
    gpus = GPUtil.getGPUs()
    gpu_usage = gpus[0].memoryUsed if gpus else "N/A"
    print(f"System Info - CPU: {cpu_usage}%, RAM: {ram_usage}%, GPU Memory: {gpu_usage}MB")

def train_type1_model1(dataset_path, disc_masks_path):
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

    start_time = time.time()
    train_losses, val_losses, train_dice_scores, val_dice_scores = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, CONFIG['num_epochs'], 'best_disc_model.pth'
    )
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    plot_curves(train_losses, val_losses, train_dice_scores, val_dice_scores, 'disc_model_curves.png')

    model.load_state_dict(torch.load('best_disc_model.pth'))
    accuracy, jaccard, dice = evaluate_model(model, test_loader, device)
    print(f"Disc Model Test Results - Accuracy: {accuracy:.4f}, Jaccard: {jaccard:.4f}, Dice: {dice:.4f}")

    return model

def train_type1_model2(disc_model, dataset_path, cup_masks_path):
    # Similar structure to train_type1_model1, but with preprocessing using disc_model
    # ...

def train_type2_model(disc_model, dataset_path, cup_masks_path):
    # Similar structure to train_type1_model1, but with transfer learning from disc_model
    # ...

def main():
    print("Starting Glaucoma Segmentation Training")
    print("Configuration:", CONFIG)
    print_system_info()

    # Train Type 1 Model 1 (Disc Segmentation)
    print("\nTraining Disc Segmentation Model")
    disc_model = train_type1_model1(CONFIG['dataset_path'], CONFIG['disc_masks_path'])

    # Train Type 1 Model 2 (Cup Segmentation)
    print("\nTraining Cup Segmentation Model (Type 1)")
    cup_model_type1 = train_type1_model2(disc_model, CONFIG['dataset_path'], CONFIG['cup_masks_path'])

    # Train Type 2 Model (Cup Segmentation with Transfer Learning)
    print("\nTraining Cup Segmentation Model (Type 2)")
    cup_model_type2 = train_type2_model(disc_model, CONFIG['dataset_path'], CONFIG['cup_masks_path'])

    print("\nTraining Complete")
    print_system_info()

if __name__ == "__main__":
    main()
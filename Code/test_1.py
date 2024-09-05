import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import accuracy_score, jaccard_score

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

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, image_size=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        img = img.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

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

def extract_largest_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    return mask

def mask_image(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_path):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_dice_scores = []
    val_dice_scores = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_dice += dice_coefficient(outputs, masks).item() * images.size(0)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                val_dice += dice_coefficient(outputs, masks).item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        train_dice = train_dice / len(train_loader.dataset)
        val_dice = val_dice / len(val_loader.dataset)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dice_scores.append(train_dice)
        val_dice_scores.append(val_dice)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    return train_losses, val_losses, train_dice_scores, val_dice_scores

def dice_coefficient(pred, target):
    smooth = 1e-5
    num = pred.size(0)
    m1 = pred.view(num, -1)
    m2 = target.view(num, -1)
    intersection = (m1 * m2).sum(1)
    return (2. * intersection + smooth) / (m1.sum(1) + m2.sum(1) + smooth)

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

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())
    
    accuracy = accuracy_score(all_targets, all_preds)
    jaccard = jaccard_score(all_targets, all_preds, average='weighted')
    dice = np.mean([dice_coefficient(torch.tensor(preds), torch.tensor(targets)).item() for preds, targets in zip(all_preds, all_targets)])
    
    return accuracy, jaccard, dice

def train_type1_model1(dataset_path, disc_masks_path, image_size=(256, 256), batch_size=32, num_epochs=1, learning_rate=0.001):
    image_paths, mask_paths = load_data(dataset_path, disc_masks_path)
    train_image_paths, temp_image_paths, train_mask_paths, temp_mask_paths = train_test_split(image_paths, mask_paths, test_size=0.3, random_state=42)
    val_image_paths, test_image_paths, val_mask_paths, test_mask_paths = train_test_split(temp_image_paths, temp_mask_paths, test_size=0.5, random_state=42)

    train_dataset = CustomDataset(train_image_paths, train_mask_paths, transform=ToTensor(), image_size=image_size)
    val_dataset = CustomDataset(val_image_paths, val_mask_paths, transform=ToTensor(), image_size=image_size)
    test_dataset = CustomDataset(test_image_paths, test_mask_paths, transform=ToTensor(), image_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model = UNet(n_classes=2, in_channels=3)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train_losses, val_losses, train_dice_scores, val_dice_scores = train_model(
    #     model, train_loader, val_loader, criterion, optimizer, device, num_epochs, 'best_disc_model.pth'
    # )
    model = UNet(n_classes=2, in_channels=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses, train_dice_scores, val_dice_scores = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs, 'best_disc_model.pth'
    )
    plot_curves(train_losses, val_losses, train_dice_scores, val_dice_scores, 'disc_model_curves.png')

    model.load_state_dict(torch.load('best_disc_model.pth'))
    accuracy, jaccard, dice = evaluate_model(model, test_loader, device)
    print(f"Disc Model Test Results - Accuracy: {accuracy:.4f}, Jaccard: {jaccard:.4f}, Dice: {dice:.4f}")

    return model

def train_type1_model2(disc_model, dataset_path, cup_masks_path, image_size=(256, 256), batch_size=32, num_epochs=1, learning_rate=0.001):
    image_paths, mask_paths = load_data(dataset_path, cup_masks_path)
    train_image_paths, temp_image_paths, train_mask_paths, temp_mask_paths = train_test_split(image_paths, mask_paths, test_size=0.3, random_state=42)
    val_image_paths, test_image_paths, val_mask_paths, test_mask_paths = train_test_split(temp_image_paths, temp_mask_paths, test_size=0.5, random_state=42)

    def preprocess_image(image_path, disc_model, device):
        image = Image.open(image_path).convert('RGB')
        image = image.resize(image_size, Image.BILINEAR)
        image_tensor = ToTensor()(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            disc_mask = torch.argmax(disc_model(image_tensor), dim=1).squeeze().cpu().numpy()
        
        disc_area = np.sum(disc_mask)
        kernel = np.ones((5, 5), np.uint8)
        dilated_disc_mask = cv2.dilate(disc_mask.astype(np.uint8), kernel, iterations=1)
        
        masked_image = mask_image(np.array(image), dilated_disc_mask)
        return ToTensor()(masked_image)

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
            mask = mask.resize(image_size, Image.NEAREST)
            mask = ToTensor()(mask).squeeze(0).long()
            return img, mask

    train_dataset = PreprocessedDataset(train_image_paths, train_mask_paths, disc_model, device)
    val_dataset = PreprocessedDataset(val_image_paths, val_mask_paths, disc_model, device)
    test_dataset = PreprocessedDataset(test_image_paths, test_mask_paths, disc_model, device)


#=========================================================================================================================================
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = UNet(n_classes=2, in_channels=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses, train_dice_scores, val_dice_scores = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs, 'best_cup_model.pth'
    )

    plot_curves(train_losses, val_losses, train_dice_scores, val_dice_scores, 'cup_model_curves.png')

    model.load_state_dict(torch.load('best_cup_model.pth'))
    accuracy, jaccard, dice = evaluate_model(model, test_loader, device)
    print(f"Cup Model Test Results - Accuracy: {accuracy:.4f}, Jaccard: {jaccard:.4f}, Dice: {dice:.4f}")

    return model

def train_type2_model(disc_model, dataset_path, cup_masks_path, image_size=(256, 256), batch_size=2, num_epochs=1, learning_rate=0.001):
    image_paths, mask_paths = load_data(dataset_path, cup_masks_path)
    train_image_paths, temp_image_paths, train_mask_paths, temp_mask_paths = train_test_split(image_paths, mask_paths, test_size=0.3, random_state=42)
    val_image_paths, test_image_paths, val_mask_paths, test_mask_paths = train_test_split(temp_image_paths, temp_mask_paths, test_size=0.5, random_state=42)

    train_dataset = CustomDataset(train_image_paths, train_mask_paths, transform=ToTensor(), image_size=image_size)
    val_dataset = CustomDataset(val_image_paths, val_mask_paths, transform=ToTensor(), image_size=image_size)
    test_dataset = CustomDataset(test_image_paths, test_mask_paths, transform=ToTensor(), image_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Copy the disc model and modify the final layer for cup segmentation
    cup_model = UNet(n_classes=2, in_channels=3)
    cup_model.load_state_dict(disc_model.state_dict())
    cup_model.final = nn.Conv2d(64, 2, 1)  # Modify the final layer for binary segmentation
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cup_model = cup_model.to(device)

    # Freeze all layers except the final layer
    for param in cup_model.parameters():
        param.requires_grad = False
    for param in cup_model.final.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cup_model.final.parameters(), lr=learning_rate)

    train_losses, val_losses, train_dice_scores, val_dice_scores = train_model(
        cup_model, train_loader, val_loader, criterion, optimizer, device, num_epochs, 'best_cup_model_type2.pth'
    )

    plot_curves(train_losses, val_losses, train_dice_scores, val_dice_scores, 'cup_model_type2_curves.png')

    cup_model.load_state_dict(torch.load('best_cup_model_type2.pth'))
    accuracy, jaccard, dice = evaluate_model(cup_model, test_loader, device)
    print(f"Cup Model (Type 2) Test Results - Accuracy: {accuracy:.4f}, Jaccard: {jaccard:.4f}, Dice: {dice:.4f}")

    return cup_model


def dice_coefficient(pred, target):
    smooth = 1e-5
    num = pred.size(0)
    pred = torch.argmax(pred, dim=1)  # Convert from one-hot to class indices
    m1 = pred.view(num, -1).float()
    m2 = target.view(num, -1).float()
    intersection = (m1 * m2).sum(1)
    return (2. * intersection + smooth) / (m1.sum(1) + m2.sum(1) + smooth)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_path):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_dice_scores = []
    val_dice_scores = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            # Add debugging information
            print(f"Output shape: {outputs.shape}")
            print(f"Mask shape: {masks.shape}")
            print(f"Output unique values: {torch.unique(outputs)}")
            print(f"Mask unique values: {torch.unique(masks)}")
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_dice += dice_coefficient(outputs, masks).mean().item() * images.size(0)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                val_dice += dice_coefficient(outputs, masks).mean().item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        train_dice = train_dice / len(train_loader.dataset)
        val_dice = val_dice / len(val_loader.dataset)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dice_scores.append(train_dice)
        val_dice_scores.append(val_dice)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    return train_losses, val_losses, train_dice_scores, val_dice_scores

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, image_size=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        img = img.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        mask = mask.squeeze(0).long()

        return img, mask

def main():
    dataset_path = "/scratch/cs23m105/Imges_Multi_model/t1/1.0_Original_Fundus_Images"
    disc_masks_path = "/scratch/cs23m105/Imges_Multi_model/t1/disc"
    cup_masks_path = "/scratch/cs23m105/Imges_Multi_model/t1/cup"

    # Train Type 1 Model 1 (Disc Segmentation)
    disc_model = train_type1_model1(dataset_path, disc_masks_path)

    # Train Type 1 Model 2 (Cup Segmentation)
    cup_model_type1 = train_type1_model2(disc_model, dataset_path, cup_masks_path)

    # Train Type 2 Model (Cup Segmentation with Transfer Learning)
    cup_model_type2 = train_type2_model(disc_model, dataset_path, cup_masks_path)

if __name__ == "__main__":
    main()
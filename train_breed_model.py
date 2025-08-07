import os
import csv
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch.cuda.amp as amp
from torch.amp import GradScaler, autocast
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 30
IMAGE_SIZE = 224
LEARNING_RATE = 0.001

# Dataset paths
TRAIN_CSV = 'data/train.csv'     # created from split_dataset.py
VAL_CSV = 'data/val.csv'
MODEL_SAVE_PATH = 'model/retrained_models/breed_classifier.pth'
ENCODER_SAVE_PATH = 'model/label_encoder.pkl'

# Create model directory if it doesn't exist
os.makedirs('model/retrained_models', exist_ok=True)

# Custom Dataset Class
class DogDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data=pd.read_csv(csv_file)
        self.transform = transform

        # Check columns
        if 'filepath' not in self.data.columns or 'label' not in self.data.columns:
            raise ValueError("CSV has to contain 'filepath' and 'label' columns")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['filepath']
        label = row['label']

        # Open image & convert to RGB
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, label
    
# ----- Transformations ------
# Training: Add random flip + slight rotation to increase variety
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),      # this flips 50% of images
    transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Validation: resize & normalize
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---- Training Function ----
scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if torch.cuda.is_available():
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# ---- Validation ----
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

if __name__ == "__main__":
    # ---- Load Datasets ----
    train_dataset = DogDataset(TRAIN_CSV, transform=train_transform)
    val_dataset = DogDataset(VAL_CSV, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # ---- Encode labels ----
    # Extract all labels from train dataset for encoding
    labels = train_dataset.data['label']
    encoder = LabelEncoder()
    encoder.fit(labels)

    # Save encoder label for future referencing
    joblib.dump(encoder, ENCODER_SAVE_PATH)
    print(f"Save label encoder to {ENCODER_SAVE_PATH}")

    # ---- Model Setup & Development ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # Load ResNet18 with pretrained weights
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4 + FC head
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True

    # Replace final fully connected layer with number of dog breed classes
    num_classes = len(encoder.classes_)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes))

    # Move model to device
    model = model.to(device)

    # Define loss function, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # ---- Main Training Loop ----
    best_val_loss = float('inf')
    patience = 3
    counter = 0
    
    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    # ---- Save Model ----
    # Check if val_loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("✅ Saved best model, training complete!")
        else:
            counter += 1
            print(f"⏳ No improvement, counter {counter}/{patience}")

            if counter >= patience:
                print("❌ Early stopping triggered!")
                break  
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 60
IMAGE_SIZE = 224
LEARNING_RATE = 0.02

# Dataset paths
TRAIN_CSV = 'data/train.csv'     # created from split_dataset.py
VAL_CSV = 'data/val.csv'
MODEL_SAVE_PATH = 'model/retrained_models/breed_classifier.pth'
ENCODER_SAVE_PATH = 'model/label_encoder.pkl'

# Create model directory if it doesn't exist
os.makedirs('model/retrained_models', exist_ok=True)

# Custom Dataset Class
class DogDataset(Dataset):
    def __init__(self, csv_file, label_encoder=None, transform=None):
        self.data=pd.read_csv(csv_file)
        self.encoder = label_encoder
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

        # string label → integer
        if self.encoder is not None:
            label = self.encoder.transform([label])[0]

        # Open image & convert to RGB
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, label
    
# ----- Transformations ------
# Training: Add random flip + slight rotation to increase variety
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),      # this flips 50% of images
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3),
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
scaler = GradScaler() if torch.cuda.is_available() else None

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1-lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()

        use_mixup = torch.rand(1).item() < 0.3

        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels)

        if torch.cuda.is_available():
            with autocast(device_type='cuda'):
                outputs = model(images)
                if use_mixup:
                    loss = lam * criterion(outputs, labels_a) + (1-lam) * criterion(outputs, labels_b)
                else:
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if use_mixup:
                loss = lam * criterion(outputs, labels_a) + (1-lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)

        if use_mixup:
            correct += (lam * (preds == labels_a).sum().item() + (1-lam) * (preds == labels_b).sum().item())
        else:
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
            labels = labels.to(device).long()

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
     # ---- Encode labels ----
    # Extract all labels from train dataset for encoding
    labels = pd.read_csv(TRAIN_CSV)['label']
    encoder = LabelEncoder()
    encoder.fit(labels)

    # Save encoder label for future referencing
    joblib.dump(encoder, ENCODER_SAVE_PATH)
    print(f"Save label encoder to {ENCODER_SAVE_PATH}")

    # ---- Load Datasets ----
    train_dataset = DogDataset(TRAIN_CSV, label_encoder=encoder, transform=train_transform)
    val_dataset = DogDataset(VAL_CSV, label_encoder=encoder, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---- Model Setup & Development ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # Load ResNet50 with pretrained weights
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4 + FC head
    for name, param in model.named_parameters():
        if any(layer in name for layer in ["layer1", "layer2", "layer3", "layer4", "fc"]):
            param.requires_grad = True

    # Replace final fully connected layer with number of dog breed classes
    num_classes = len(encoder.classes_)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes))

    # Move model to device
    model = model.to(device)

    # Define loss function, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ---- Main Training Loop ----
    best_val_loss = float('inf')
    no_improve_epochs = 0
    unfreezed_layer1 = False
    best_epoch = 0
    
    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        if epoch < 5:
            warmup_lr = LEARNING_RATE * ((epoch + 1) / 5) ** 2
            for g in optimizer.param_groups:
                 g['lr'] = warmup_lr
        else:
            scheduler.step()

        if epoch == 30:
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
            scheduler=CosineAnnealingLR(optimizer, T_max=EPOCHS - 30)
            print("🔄 Switched to AdamW for final fine-tuning from epoch 30!")
        
        if epoch == EPOCHS - 1:  # final epoch
            for g in optimizer.param_groups:
                g['lr'] *= 0.1

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # ---- Save Model ----
        # Check if val_loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Saved best model at epoch {epoch+1} (training checkpoint)")
        else:
            no_improve_epochs += 1
            print("⏳ No improvement in val_loss")

        if no_improve_epochs >= 5 and not unfreezed_layer1:
            for p in model.layer1.parameters():
                p.requires_grad = True
            unfreezed_layer1 = True
            print("🔓 Unfroze layer1 for fine-tuning")

    print(f"Training complete. Best Validation Loss: {best_val_loss:.4f}")

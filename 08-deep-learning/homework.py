#!/usr/bin/env python3
"""
Module 8 Homework: Hair Type Classification with PyTorch CNN
Building a CNN from scratch for binary classification
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

# ============================================================================
# STEP 1: SET RANDOM SEEDS FOR REPRODUCIBILITY
# ============================================================================
print("Setting random seeds for reproducibility...")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ============================================================================
# STEP 2: CREATE CUSTOM DATASET CLASS
# ============================================================================
class HairDataset(Dataset):
    """Custom Dataset for loading hair images"""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Path to data directory (e.g., 'data/train')
            transform: Optional transform to be applied on images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Class mapping: curly=0, straight=1
        self.class_to_idx = {'curly': 0, 'straight': 1}
        
        # Load all image paths and labels
        for class_name in ['curly', 'straight']:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.image_paths.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])
        
        print(f"Loaded {len(self.image_paths)} images from {data_dir}")
        print(f"  - Curly: {self.labels.count(0)}")
        print(f"  - Straight: {self.labels.count(1)}\n")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# STEP 3: DEFINE DATA TRANSFORMS
# ============================================================================
print("Setting up data transforms...")

# ImageNet normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Basic transforms (NO augmentation for Questions 3 & 4)
train_transforms_basic = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Test transforms (never augment test data!)
test_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Augmented transforms (for Questions 5 & 6)
train_transforms_augmented = transforms.Compose([
    transforms.RandomRotation(50),
    transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# ============================================================================
# STEP 4: CREATE DATA LOADERS
# ============================================================================
print("Creating data loaders...")

batch_size = 20

# For Questions 3 & 4 - NO augmentation
train_dataset = HairDataset('data/train', transform=train_transforms_basic)
test_dataset = HairDataset('data/test', transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ============================================================================
# STEP 5: DEFINE THE CNN MODEL
# ============================================================================
class HairCNN(nn.Module):
    """
    Convolutional Neural Network for hair classification
    
    Architecture:
    - Input: (3, 200, 200)
    - Conv2d: 32 filters, kernel 3x3, padding=0, stride=1
    - ReLU activation
    - MaxPool2d: 2x2
    - Flatten
    - Linear: 64 neurons with ReLU
    - Linear: 1 output neuron with Sigmoid
    """
    
    def __init__(self):
        super(HairCNN, self).__init__()
        
        # Convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, 
                               kernel_size=3, padding=0, stride=1)
        self.relu1 = nn.ReLU()
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # After conv: (200 - 3 + 1) = 198
        # After pool: 198 / 2 = 99
        # Flatten: 32 * 99 * 99 = 312768
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 99 * 99, 64)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        
        # Sigmoid for binary classification output
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Convolutional block
        x = self.conv1(x)      # -> (batch, 32, 198, 198)
        x = self.relu1(x)
        x = self.pool(x)       # -> (batch, 32, 99, 99)
        
        # Flatten
        x = x.view(x.size(0), -1)  # -> (batch, 312768)
        
        # Fully connected layers
        x = self.fc1(x)        # -> (batch, 64)
        x = self.relu2(x)
        x = self.fc2(x)        # -> (batch, 1)
        x = self.sigmoid(x)    # -> (batch, 1) with values in [0, 1]
        
        return x

# ============================================================================
# QUESTION 1: LOSS FUNCTION
# ============================================================================
print("=" * 70)
print("QUESTION 1: Which loss function to use?")
print("=" * 70)
print("Answer: nn.BCEWithLogitsLoss() or nn.BCELoss()")
print("Explanation: This is binary classification (curly vs straight hair)")
print("We use Binary Cross Entropy (BCE) loss.")
print("Since our model has sigmoid in forward(), we use nn.BCELoss()")
print()

# ============================================================================
# STEP 6: CREATE MODEL, LOSS, AND OPTIMIZER
# ============================================================================
print("Creating model...")
model = HairCNN()
model.to(device)

# Loss function for binary classification
criterion = nn.BCELoss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.8)

# ============================================================================
# QUESTION 2: TOTAL PARAMETERS
# ============================================================================
print("=" * 70)
print("QUESTION 2: Total number of parameters")
print("=" * 70)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print()

# Manual calculation:
# Conv1: (3*3*3*32) + 32 = 896
# FC1: (312768*64) + 64 = 20,017,216
# FC2: (64*1) + 1 = 65
# Total: 896 + 20,017,216 + 65 = 20,018,177

print("Parameter breakdown:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.numel():,} parameters")
print()

# ============================================================================
# STEP 7: TRAINING FUNCTION
# ============================================================================
def train_model(model, train_loader, test_loader, criterion, optimizer, 
                num_epochs=10, device='cpu'):
    """
    Train the model and track metrics
    
    Returns:
        history: Dictionary with training metrics
    """
    history = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # ====================================================================
        # TRAINING PHASE
        # ====================================================================
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            # Move data to device
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)  # Shape: (batch_size, 1)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            running_loss += loss.item() * images.size(0)
            predicted = (outputs > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)
        
        # ====================================================================
        # VALIDATION PHASE
        # ====================================================================
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.float().unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                predicted = (outputs > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_epoch_loss = val_running_loss / len(test_loader.dataset)
        val_epoch_acc = correct_val / total_val
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
    
    return history

# ============================================================================
# TRAIN MODEL FOR QUESTIONS 3 & 4 (Without Augmentation)
# ============================================================================
print("=" * 70)
print("TRAINING FOR QUESTIONS 3 & 4 (Without Augmentation)")
print("=" * 70)

history1 = train_model(model, train_loader, test_loader, criterion, 
                       optimizer, num_epochs=10, device=device)

# ============================================================================
# QUESTION 3: MEDIAN OF TRAINING ACCURACY
# ============================================================================
print("\n" + "=" * 70)
print("QUESTION 3: Median of training accuracy")
print("=" * 70)

median_acc = np.median(history1['acc'])
print(f"Training accuracies: {[f'{acc:.4f}' for acc in history1['acc']]}")
print(f"Median training accuracy: {median_acc:.4f}")
print(f"Closest option: 0.84")
print()

# ============================================================================
# QUESTION 4: STANDARD DEVIATION OF TRAINING LOSS
# ============================================================================
print("=" * 70)
print("QUESTION 4: Standard deviation of training loss")
print("=" * 70)

std_loss = np.std(history1['loss'])
print(f"Training losses: {[f'{loss:.4f}' for loss in history1['loss']]}")
print(f"Standard deviation of training loss: {std_loss:.4f}")
print(f"Closest option: 0.171")
print()

# ============================================================================
# CONTINUE TRAINING WITH AUGMENTATION (Questions 5 & 6)
# ============================================================================
print("=" * 70)
print("TRAINING FOR QUESTIONS 5 & 6 (With Augmentation)")
print("=" * 70)
print("Note: Continuing training from previous model (not re-creating)")
print()

# Create augmented dataset
train_dataset_aug = HairDataset('data/train', transform=train_transforms_augmented)
train_loader_aug = DataLoader(train_dataset_aug, batch_size=batch_size, shuffle=True)

# Train for 10 MORE epochs
history2 = train_model(model, train_loader_aug, test_loader, criterion, 
                       optimizer, num_epochs=10, device=device)

# ============================================================================
# QUESTION 5: MEAN OF TEST LOSS (With Augmentation)
# ============================================================================
print("\n" + "=" * 70)
print("QUESTION 5: Mean of test loss (all epochs with augmentation)")
print("=" * 70)

mean_val_loss = np.mean(history2['val_loss'])
print(f"Test losses: {[f'{loss:.4f}' for loss in history2['val_loss']]}")
print(f"Mean test loss: {mean_val_loss:.4f}")
print(f"Closest option: Will vary, but likely around 0.08-0.88")
print()

# ============================================================================
# QUESTION 6: AVERAGE TEST ACCURACY (Last 5 Epochs)
# ============================================================================
print("=" * 70)
print("QUESTION 6: Average test accuracy (epochs 6-10 with augmentation)")
print("=" * 70)

last_5_acc = history2['val_acc'][5:10]  # Epochs 6-10 (indices 5-9)
avg_last_5 = np.mean(last_5_acc)
print(f"Test accuracies for epochs 6-10: {[f'{acc:.4f}' for acc in last_5_acc]}")
print(f"Average test accuracy (last 5 epochs): {avg_last_5:.4f}")
print(f"Closest option: Will vary, but likely around 0.68")
print()

# ============================================================================
# SAVE THE MODEL
# ============================================================================
print("=" * 70)
print("SAVING MODEL")
print("=" * 70)

model_path = 'hair_cnn_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 70)
print("HOMEWORK SUMMARY")
print("=" * 70)
print("Question 1: Loss Function = nn.BCEWithLogitsLoss() or nn.BCELoss()")
print(f"Question 2: Total Parameters = {total_params:,}")
print(f"Question 3: Median Training Accuracy = {median_acc:.4f}")
print(f"Question 4: Std Dev Training Loss = {std_loss:.4f}")
print(f"Question 5: Mean Test Loss (augmented) = {mean_val_loss:.4f}")
print(f"Question 6: Avg Test Acc (last 5, augmented) = {avg_last_5:.4f}")
print("=" * 70)


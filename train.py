import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import time
import os
from tqdm import tqdm

# 1. SETUP
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 3
DATA_PATH = './data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# 2. DATA PREPARATION (With Augmentation)
# We add RandomHorizontalFlip and RandomRotation to make the model robust.
# We also normalize using the standard ImageNet mean/std values.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the entire dataset
full_dataset = datasets.ImageFolder(DATA_PATH, transform=transform)

# Split into Train (80%) and Validation (20%)
# This ensures we test on images the model has NEVER seen.
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training on {train_size} images, Validating on {val_size} images.")

# 3. SETUP MODEL (Transfer Learning)
# Download pre-trained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all layers (so we don't mess up the pre-trained 'brain')
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer (The 'Head')
# ResNet18's original fully connected layer (fc) has 512 inputs and 1000 outputs (ImageNet classes).
# We change it to output 2 classes: [Negative, Positive]
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model = model.to(DEVICE)

# 4. LOSS AND OPTIMIZER
criterion = nn.CrossEntropyLoss()
# Only optimize the parameters of the final layer (model.fc)
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# 5. TRAINING LOOP
print("\nStarting Training...")
start_time = time.time() 
    # Training Phase
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # WRAP THE LOADER WITH TQDM
    # This creates the progress bar object
    loop = tqdm(train_loader, leave=True)
    loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
    
    for images, labels in loop:   # <--- USE 'loop' INSTEAD OF 'train_loader'
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # UPDATE THE BAR
        # This updates the text next to the bar in real-time
        loop.set_postfix(loss=loss.item(), acc=100*correct/total)
    train_acc = 100 * correct / total
    # ... (keep validation code the same) ...
    
    # Validation Phase (Check accuracy on unseen data)
    model.eval() # Set model to evaluation mode
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_acc = 100 * val_correct / val_total
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {running_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Val Acc: {val_acc:.2f}%")

total_time = time.time() - start_time
print(f"\nTraining Finished in {total_time/60:.2f} minutes.")

# 6. SAVE MODEL
torch.save(model.state_dict(), 'crack_detection_model.pth')
print("Model saved to crack_detection_model.pth")
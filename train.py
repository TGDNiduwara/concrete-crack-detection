import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import time
import os
from tqdm import tqdm
import mlflow # <--- NEW IMPORT
import mlflow.pytorch

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

# 2. SETUP MODEL
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# --- START MLFLOW EXPERIMENT ---
mlflow.set_experiment("Concrete Crack Detection")

with mlflow.start_run(): # <--- EVERYTHING RUNS INSIDE THIS BLOCK
    
    # Log your Hyperparameters (So you remember what settings you used)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("model", "ResNet18")

    print("\nStarting Training...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
        
        for images, labels in loop:
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
            
            loop.set_postfix(loss=loss.item(), acc=100*correct/total)
        
        # Calculate Epoch Metrics
        epoch_loss = running_loss/len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation Phase
        model.eval()
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
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # --- LOG METRICS TO MLFLOW ---
        mlflow.log_metric("loss", epoch_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)

    # Save the model to MLflow (and locally)
    mlflow.pytorch.log_model(model, "model")
    torch.save(model.state_dict(), 'crack_detection_model.pth')
    print("Training Finished and logged to MLflow!")
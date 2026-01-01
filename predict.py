import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os
import random

# 1. SETUP
DATA_PATH = './data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ['Negative', 'Positive'] # Negative = No Crack, Positive = Crack

# 2. RE-BUILD THE MODEL ARCHITECTURE
# (We must define the exact same structure to load the weights)
model = models.resnet18(weights=None) # No need to download ImageNet weights again
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Load the trained weights
model.load_state_dict(torch.load('crack_detection_model.pth'))
model.to(DEVICE)
model.eval()
print("✅ Model loaded successfully!")

# 3. PREPARE DATA TRANSFORM
# Must match the training transform (Resize + Normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset just to pick random samples
# (We use ImageFolder again because it's the easiest way to grab files)
dataset = datasets.ImageFolder(DATA_PATH, transform=transform)

# 4. PREDICT LOOP (Show 3 random examples)
def show_prediction():
    # Pick a random index
    idx = random.randint(0, len(dataset)-1)
    image_tensor, label_idx = dataset[idx]
    
    # Add batch dimension [3, 224, 224] -> [1, 3, 224, 224]
    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        _, pred_idx = torch.max(output, 1)
        
    prob = torch.nn.functional.softmax(output, dim=1)[0] * 100
    
    # Display
    actual_class = CLASSES[label_idx]
    predicted_class = CLASSES[pred_idx.item()]
    confidence = prob[pred_idx.item()].item()
    
    # Un-normalize for display (so colors look normal)
    # Undo: image = (image - mean) / std  ->  image = image * std + mean
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_display = image_tensor.cpu() * std + mean
    img_display = torch.clamp(img_display, 0, 1) # Clip values to be safe
    
    plt.imshow(img_display.permute(1, 2, 0))
    plt.title(f"Actual: {actual_class}\nPred: {predicted_class} ({confidence:.1f}%)")
    plt.axis('off')
    plt.show()

# Run it
print("Displaying random prediction...")
show_prediction()
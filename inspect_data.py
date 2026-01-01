import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# 1. SETUP PATHS
DATA_PATH = './data'

# Check if data exists
if not os.path.exists(DATA_PATH):
    print(f"❌ Error: Could not find {DATA_PATH}. Did you unzip the dataset there?")
    exit()

# 2. DEFINE TRANSFORMS (Resize is important for real photos!)
# ResNet expects 224x224 images.
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 3. LOAD DATA
try:
    dataset = datasets.ImageFolder(root=DATA_PATH, transform=data_transform)
    print(f"✅ Dataset loaded successfully!")
    print(f"   Total images: {len(dataset)}")
    print(f"   Classes found: {dataset.classes}") # Should be ['Negative', 'Positive']
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# 4. VISUALIZE A FEW SAMPLES
# We want to see what a "Positive" (Crack) looks like vs "Negative"
fig, axes = plt.subplots(1, 4, figsize=(12, 4))
indices = [0, 1, 20001, 20002] # Picking specific indices (assuming roughly 20k each)

for i, idx in enumerate(indices):
    if idx < len(dataset):
        image, label = dataset[idx]
        class_name = dataset.classes[label]
        
        # Un-normalize not needed yet since we didn't normalize, just permute for matplotlib
        # PyTorch is [Channel, Height, Width] -> Matplotlib wants [Height, Width, Channel]
        image_display = image.permute(1, 2, 0)
        
        axes[i].imshow(image_display)
        axes[i].set_title(class_name)
        axes[i].axis('off')

plt.show()
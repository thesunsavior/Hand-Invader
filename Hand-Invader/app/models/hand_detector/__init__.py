import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from app.models.pipeline import Pipeline
from image_dataset import ImageDataset

ROOT_DIR = os.path.dirname(os.path.abspath(__name__)) # This is your Project Root

# data preparation
IMAGES_DIR = ROOT_DIR+ '/Hand-Invader/app/data/EgoHands/train/images'
LABELS_DIR = ROOT_DIR+ '/Hand-Invader/app/data/EgoHands/train/labels'

image_paths = [os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")]
label_paths = [os.path.join(LABELS_DIR, f.replace(".png", ".txt")) for f in os.listdir(LABELS_DIR) if f.endswith(".jpg")]

transform = transforms.Compose([
    transforms.Resize(256),  # Resize images to 256x256 (adjust as needed)
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

images_dataset = ImageDataset(image_paths=image_paths, label_paths=label_paths, transform=transform)
images_dataloader = DataLoader(images_dataset, batch_size=16, shuffle=True) 



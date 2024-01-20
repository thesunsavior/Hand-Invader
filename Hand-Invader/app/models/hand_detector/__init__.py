import os
import json

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from app.models.pipeline import Pipeline
from image_dataset import ImageDataset

ROOT_DIR = os.path.dirname(os.path.abspath(__name__)) # This is your Project Root

# data preparation
IMAGES_DIR = ROOT_DIR+ '/Hand-Invader/app/data/EgoHands/train/'
LABELS_JSON = ROOT_DIR+ '/Hand-Invader/app/data/EgoHands/train/_annotations.coco.json'

with open(LABELS_JSON, "r") as f:
    coco_data = json.load(f)

transform = transforms.Compose([
    transforms.Resize(256),  # Resize images to 256x256 (adjust as needed)
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

images_dataset = ImageDataset(image_dir= IMAGES_DIR, coco_json= coco_data, transform=transform)
images_dataloader = DataLoader(images_dataset, batch_size=16, shuffle=True) 


# model pipeline 

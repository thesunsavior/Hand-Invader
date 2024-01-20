import os
import json

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from app.models.pipeline import Pipeline
from app.models.hand_detector.image_dataset import ImageDataset

def collate_fn(batch):
  return tuple(zip(*batch))

ROOT_DIR = os.path.dirname(os.path.abspath(__name__)) # This is your Project Root

# data preparation
IMAGES_DIR = ROOT_DIR+ '/app/data/EgoHands/train/'
LABELS_JSON = ROOT_DIR+'/app/data/EgoHands/train/_annotations.coco.json'

with open(LABELS_JSON, "r") as f:
    coco_data = json.load(f)

transform = transforms.Compose([
    transforms.Resize(256),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

images_dataset = ImageDataset(image_dir= IMAGES_DIR, coco_json= coco_data, 
                              transform=transform)
images_dataloader = DataLoader(images_dataset, batch_size=1, 
                               shuffle=False, num_workers=2,collate_fn=collate_fn) 

# model pipeline 
hand_detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# replace last layer
num_classes = 22
in_features = hand_detection_model.roi_heads.box_predictor.cls_score.in_features

hand_detection_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

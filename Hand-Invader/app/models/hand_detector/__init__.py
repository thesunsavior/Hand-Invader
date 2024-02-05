import os
import json

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.transforms import v2 as T

from app.models.hand_detector.image_dataset import ImageDataset

def collate_fn(batch):
  return tuple(zip(*batch))

def get_transform(train=False):
    transformss = [transforms.ToTensor(), transforms.Resize(320)]
    if train:
        transformss.append(T.RandomHorizontalFlip(0.5))
    transformss.append(T.ToDtype(torch.float, scale=True))
    transformss.append(T.ToPureTensor())
    return T.Compose(transformss)

def create_model(num_classes=2, size=256):
    # model pipeline 
    model = torchvision.models.detection.ssd300_vgg16(num_classes=num_classes, weights_backbone='DEFAULT', trainable_backbone_layers=1)

    return model

ROOT_DIR = os.path.dirname(os.path.abspath(__name__)) # This is your Project Root

# data preparation
IMAGES_DIR = ROOT_DIR+ '/app/data/EgoHands/train/'
LABELS_JSON = ROOT_DIR+'/app/data/EgoHands/train/_annotations.coco.json'

IMAGES_VAL_DIR = ROOT_DIR+ '/app/data/EgoHands/valid/'
LABELS_VAL_JSON = ROOT_DIR+'/app/data/EgoHands/valid/_annotations.coco.json'

with open(LABELS_JSON, "r") as f:
    coco_data = json.load(f)

with open(LABELS_VAL_JSON, "r") as f:
    coco_valid_data = json.load(f)


images_dataset = ImageDataset(image_dir= IMAGES_DIR, coco_json= coco_data, 
                              transform=get_transform())
images_dataloader = DataLoader(images_dataset, batch_size=32, 
                               shuffle=False, num_workers=2,collate_fn=collate_fn) 

images_valid_dataset = ImageDataset(image_dir= IMAGES_VAL_DIR, coco_json= coco_valid_data, 
                              transform=get_transform())
images_valid_dataloader = DataLoader(images_valid_dataset, batch_size=32, 
                               shuffle=False, num_workers=2,collate_fn=collate_fn) 

hand_detection_model = create_model(num_classes=2, size=300)

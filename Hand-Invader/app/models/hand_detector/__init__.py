import os
import json

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from torchvision.models.detection import SSD300_VGG16_Weights

from app.models.pipeline import Pipeline
from app.models.hand_detector.image_dataset import ImageDataset

def collate_fn(batch):
  return tuple(zip(*batch))

def get_transform(train=False):
    transformss = [transforms.Resize(300), transforms.ToTensor()]
    if train:
        transformss.append(T.RandomHorizontalFlip(0.5))
    transformss.append(T.ToDtype(torch.float, scale=True))
    transformss.append(T.ToPureTensor())
    return T.Compose(transformss)

def create_model(num_classes=2, size=256):
    # model pipeline 
    model = torchvision.models.detection.ssd300_vgg16(
        weights=SSD300_VGG16_Weights.COCO_V1
    )

    for param in model.parameters():
        param.requires_grad = False

    # Retrieve the list of input channels. 
    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))

    # List containing number of anchors based on aspect ratios.
    num_anchors = model.anchor_generator.num_anchors_per_location()
    # The classification head.
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )

    return model

ROOT_DIR = os.path.dirname(os.path.abspath(__name__)) # This is your Project Root

# data preparation
IMAGES_DIR = ROOT_DIR+ '/app/data/EgoHands/train/'
LABELS_JSON = ROOT_DIR+'/app/data/EgoHands/train/_annotations.coco.json'

with open(LABELS_JSON, "r") as f:
    coco_data = json.load(f)


images_dataset = ImageDataset(image_dir= IMAGES_DIR, coco_json= coco_data, 
                              transform=get_transform())
images_dataloader = DataLoader(images_dataset, batch_size=1, 
                               shuffle=False, num_workers=2,collate_fn=collate_fn) 

hand_detection_model = create_model(num_classes=2, size=300)

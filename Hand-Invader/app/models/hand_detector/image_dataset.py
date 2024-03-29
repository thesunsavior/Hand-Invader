import cv2 as cv

import torch
from torchvision import tv_tensors

from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, image_dir, coco_json, transform=None):
        self.image_dir = image_dir
        self.coco_json = coco_json
        self.transform = transform

        self.image_paths = []
        for idx, image in enumerate(coco_json['images']):
            assert idx == image['id']
            self.image_paths.append(image_dir+"/"+image['file_name'])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Read image and label
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        target = {}
        boxes =[]
        labels=[]
        for  annotated_imaged in self.coco_json['annotations']:
            if annotated_imaged['id'] == idx:
                # convert XYWH to XYXY
                bbox_XYXY = annotated_imaged['bbox']
                bbox_XYXY = [bbox_XYXY[0], bbox_XYXY[1], bbox_XYXY[0]+bbox_XYXY[2], 
                             bbox_XYXY[1]+bbox_XYXY[3]]
                boxes.append(bbox_XYXY)
                labels.append(1)

        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=320)
        target["labels"] = torch.tensor(labels)

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        return image, target
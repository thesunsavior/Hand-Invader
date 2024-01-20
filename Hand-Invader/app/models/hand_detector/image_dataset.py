import torch
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, image_dir, coco_json, transform=None):
        self.image_dir = image_dir
        self.coco_json = coco_json
        self.transform = transform

        self.image_paths = []
        for idx, image in enumerate(coco_json['images']):
            assert idx == image['id']
            self.image_paths.append(image['files_name'])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Read image and label
        image = Image.open(image_path)
        
        labels = []
        for  annotated_imaged in self.coco_json['annotations']:
            if annotated_imaged['id'] == idx:
                labels.append(annotated_imaged['bbox'])

        labels_tensor = torch.tensor(labels)

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        return image, labels_tensor
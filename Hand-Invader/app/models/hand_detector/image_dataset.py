import torch
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Read image and label
        image = Image.open(image_path)
        label = torch.tensor(int(open(label_path, "r").readline()))  # Assuming labels are single integers

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        return image, label
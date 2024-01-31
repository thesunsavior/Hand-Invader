from app.models.hand_detector import hand_detection_model, images_dataloader, get_transform
from app.util.train_util import train_hand_detection
from app.util.eval_util import inference, plot_image, plot_loss

import torch

from PIL import Image


if __name__ == "__main__":
    train_hand_detection(hand_detection_model, 5)
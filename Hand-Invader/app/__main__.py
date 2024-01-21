import app.models
from app.models.hand_detector import hand_detection_model, transform, images_dataloader
from app.models.pipeline import train_one_epoch, save_checkpoint
from app.models.hand_detector.util import inference, plot_image

import torch

from PIL import Image


if __name__ == "__main__":
     # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # move model to the right device
    hand_detection_model.to(device)

    # construct an optimizer
    params = [p for p in hand_detection_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # let's train it just for 2 epochs
    num_epochs = 2

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(hand_detection_model, optimizer, images_dataloader, device)
        # update the learning rate
        lr_scheduler.step()
    
    save_checkpoint(hand_detection_model, optimizer, 'hand_detection.ckpt')

    # prediction
    image_path = '/Users/trungpham/Public/Hand-Invader/Hand-Invader/Y-I_mp4-48_jpg.rf.8a8f67f98b27b4543c35b43668d4532a.jpg'
    image = Image.open(image_path)

    # Convert the PIL image to Torch tensor 
    img_tensor = transform(image) 

    boxes, score, label = inference(img_tensor, hand_detection_model, 'cpu',0)
    
    plot_image(img_tensor,boxes=boxes, scores=score, labels=label, dataset=[x for x in range(21)])
    
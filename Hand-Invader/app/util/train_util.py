import cv2 as cv

import torch
from PIL import Image

from app.models.pipeline import train_one_epoch, save_checkpoint, load_checkpoint
from app.models.hand_detector import images_dataloader, get_transform
from app.util.eval_util import inference, plot_image, plot_loss


def train(num_epochs, model, device, optimizer, lr_scheduler):
    train_losses =[]
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_losses.append(train_one_epoch(model, optimizer, images_dataloader, device))
        # update the learning rate
        lr_scheduler.step()
    
        save_checkpoint(model, optimizer, train_losses,'hand_detection.ckpt')
    
    return train_losses

def train_hand_detection(hand_detection_model, num_epochs,resume= False, ckpt_filename=""):
    # Set up parameter
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # move model to the right device
    hand_detection_model.to(device)

    # construct an optimizer
    params = [p for p in hand_detection_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=0.1,
        gamma=0.1
    )

    
    losses =[]

    # Resume training 
    if resume:
        hand_detection_model, optimizer, losses = load_checkpoint(model=hand_detection_model,optimizer=optimizer,filename=ckpt_filename)
    
    losses= losses + train(num_epochs, hand_detection_model, device, optimizer, lr_scheduler)

    # load ckpt
    hand_detection_model, optimizer, losses = load_checkpoint(model=hand_detection_model,optimizer=optimizer,filename=ckpt_filename)
    print(losses)

    # prediction
    image_path = 'A-D_mp4-1_jpg.rf.c2050f2692c0a255c8da67549e5617d2.jpg'
    image = Image.open(image_path).convert('RGB')

    # Convert the PIL image to Torch tensor 
    transform = get_transform()
    img_tensor = transform(image) 

    hand_detection_model.eval()
    print(hand_detection_model([img_tensor]))

    boxes, score, label = inference(img_tensor, hand_detection_model, 'cpu',0.3)
    plot_loss(losses)
    
    plot_image(img_tensor,boxes=boxes, scores=score, labels=label, dataset=[x for x in range(2)])

    return hand_detection_model

def draw_bbox(img, boxes):
    # box should be a list of form (x,y,w,h)
    img_temp = img
    for box in boxes:
        img_temp= cv.rectangle(img,(box[0],box[1]),(box[0]+box[2], box[1]+box[3]),2);

    return img_temp

import numpy as np
import torch
from tqdm import tqdm


class Pipeline:
    def __init__(self, detection_mode= "mdp", classification_mode="mdp"):
        pass
    
    

def train_one_epoch(model, optimizer, data_loader, device='cpu'):
  model.train()
  train_loss_list = []

  tqdm_bar = tqdm(data_loader, total=len(data_loader))
  for idx, data in enumerate(tqdm_bar):
    optimizer.zero_grad()
    images, targets = data

    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # targets = {'boxes'=tensor, 'labels'=tensor}

    losses = model(images, targets)

    loss = sum(loss for loss in losses.values())
    loss_val = loss.item()
    train_loss_list.append(loss_val)

    loss.backward()
    optimizer.step()

    tqdm_bar.set_description(desc=f"Training Loss: {loss:.3f}")

  return np.mean(train_loss_list)

def save_checkpoint(model, optimizer, train_losses, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    # Loads dictionary
    checkpoint = torch.load(filename)
    
    # Restore state for model and optimizer
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(
        checkpoint['optimizer_state_dict']
    )
    train_losses = checkpoint['train_loss']
    model.train() 

    return model, optimizer, train_losses
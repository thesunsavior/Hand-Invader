import numpy as np

import torch

class Pipeline:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # check if GPU available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None

        self.train_losses = []
        self.val_losses = []
        self.total_epoch = 0

        self.train_step_fn = self._make_train_step_fn
        self.val_step_fn = self._make_val_step_fn

        
    def set_loader(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
    

    def _make_train_step_fn(self):
        def train_step(X, y):
            self.model.train()

            # Forward pass
            yhat= self.model(X)

            # Calculate loss
            loss = self.loss_fn(yhat, y)
        
            # Compute grad
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            
            return loss.item()

        return train_step
    
    def _make_val_step_fn(self):
        def val_step(X, y):
            self.model.eval()

            # Forward pass
            yhat= self.model(X)

            # Calculate loss
            loss = self.loss_fn(yhat, y)
                    
            return loss.item()
        
        return val_step
    
    def _mini_batch(self, is_validation= False):
        if is_validation:
            dataloader = self.val_loader
            step_fn = self.val_step_fn
        else:
            dataloader = self.train_loader
            step_fn = self.train_step_fn

        if dataloader is None:
            return None

        batch_losses = []
        for X_batch, y_batch in dataloader:
            X_batch.to(self.device)
            y_batch.to(self.device)

            loss = step_fn(X_batch, y_batch)
            batch_losses.append(loss)

        return np.mean(batch_losses)
    
    def set_seed(self, seed=20):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train(self, n_epoch, seed=20):
        self.set_seed(seed)
        
        for epoch in range(n_epoch):
            # train
            self.total_epoch +=1

            loss= self._mini_batch(is_validation=False)
            self.train_losses.append(loss)

            # validation 
            with torch.no_grad():
                loss =  self._mini_batch(is_validation=True)
                self.val_losses.append(loss)

        # todo (Trung):  add Tensorboard writer for easy tracking
                
    def save_checkpoint(self, filename):
        checkpoint = {
            'epoch': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename):
        # Loads dictionary
        checkpoint = torch.load(filename)
        
        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict']
        )
        self.total_epochs = checkpoint['epoch']
        self.train_losses = checkpoint['train_loss']
        self.val_losses = checkpoint['val_loss']
        self.model.train() 
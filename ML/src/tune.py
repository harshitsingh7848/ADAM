import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import copy

class Tuner:
    def __init__(self, model, data_dir, patience=7):
        self.model = model # YoloThreat
        self.original_model = copy.deepcopy(model)
        self.data_dir = data_dir # ML/data/{data_dir}
        self.patience = patience
        
        self.best_model = None
        self.best_loss = np.inf
        self.patience_counter = 0

    '''
    This runs fine tuning on the model and returns the trained model
    The only non-traditional hyperparams are freeze_decay and freeze_limit
    This has to do with the layers that are frozen during training
    Freeze_limit is the layer number that is never unfrozen
    Freeze_decay controls how many layers are unfrozen at each epoch
    No matter what, all layers greater than the freeze_limit will be unfrozen at some point.
    At the start, only the last layer is unfrozen. At the end, all layers are unfrozen.
    The decay is exponential and is trucated to the nearest integer, this determines how many layers are unfrozen at each epoch
    '''
    def tune(self, optimizer = optim.SGD, epochs = 50, batch_size = 32, lr = 0.01, freeze_limit = 10, warmup = 0.1, lam = 0.01):
        Xtrain, ytrain = self.data_dir
        Xtrain, ytrain, Xval, yval = self.split_data(Xtrain, ytrain)

        train_dataset = torch.utils.data.TensorDataset(Xtrain, ytrain)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(Xval, yval)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = self.get_loss(lam)
        optimizer = optimizer(self.model.parameters(), lr=lr)

        warmup_epochs = int(epochs * warmup)

        self.model.train()

        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()  
 
                    
        for epoch in range(epochs):

            if epoch < warmup_epochs:
                freeze = self.model.upper_layer
            else:
                freeze = freeze_limit

            self.model.freeze_below(freeze)

            for i, (X_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                y_pred = self.model.forward(X_batch)
                y_pred =y_pred.squeeze(-1)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

            val_loss = 0
            for i, (X_batch, y_batch) in enumerate(val_loader):
                y_pred = self.model.forward(X_batch)
                y_pred = y_pred.squeeze(-1)
                val_loss += criterion(y_pred, y_batch).item()

            if epoch > warmup_epochs and self.early_stop(val_loss):
                print(f'Early stopping at epoch {epoch}')
                return self.best_model

            if i % 1 == 0:
                print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item()}')
                print(f'Validation Loss: {val_loss}')

        self.model.eval()
        
        return self.best_model
    
    def early_stop(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(self.model)
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.patience:
            return True
        
        return False
    
    '''
    Generate validation split
    '''
    def split_data(self, X, y):
        split = int(len(X) * 0.8)
        Xtrain, ytrain = X[:split], y[:split]
        Xval, yval = X[split:], y[split:]
        
        return Xtrain, ytrain, Xval, yval

    '''
    The loss function for the fine tuned model is a combination of the baseline loss and an L2 term.
    The L2 term is intended to keep the new model from changing the weights o the priginal pretrained model too dramatically.
    The following function is a closure which returns the desired loss with a given lambda to scale the L2 term.
    '''
    def get_loss(self, lam):

        def loss(y_pred, y_true):
            baseline = torch.nn.BCEWithLogitsLoss()

            l2 = 0
            for (old_names, old_params), (new_names, new_params) in zip(self.original_model.named_parameters(), self.model.named_parameters()):
                yolo_layer = self.model.YoloLayer(new_names, new_params)
                if yolo_layer and yolo_layer < self.model.upper_layer - 1:
                    l2 += torch.norm(new_params - old_params, p=2)

            return baseline(y_pred, y_true) + lam * l2
    
        return loss

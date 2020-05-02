import torch
from torch import nn
from torch import optim


def train_model(model, train_input, train_target, mini_batch_size):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    for e in range(25):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            optimizer.zero_grad()
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            loss.backward()
            optimizer.step()
           
        
######################################################################################

def accuracy_one_hot(preds, target):
    return (preds.argmax(axis=1) == target.argmax(axis=1)).long().sum().item() / target.shape[0]
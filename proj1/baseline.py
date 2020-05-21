import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np

class MLP(nn.Module):
    """ Modified LeNet. Takes input format 2 x 14 x 14"""
    def __init__(self):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(392, 264),
            nn.ReLU(inplace=True),
            nn.Linear(264, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 2)
)
        
    def forward(self, x):
        x = self.classifier(x)
        return x
    
    
def train(model, train_input, train_target,val_input, val_target, mini_batch_size, nb_epochs = 25,verbose = False):
    losses = np.zeros(nb_epochs)
    val_losses = np.zeros(nb_epochs)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for e in range(nb_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            optimizer.zero_grad()
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            loss.backward()
            optimizer.step()
        if verbose:
            print(e,sum_loss)
        losses[e] = sum_loss
        val_losses[e] = get_loss_val(model.eval(), val_input, val_target)
    return losses,val_losses

    
def get_loss_val(model, val_input,val_target):
    criterion = nn.CrossEntropyLoss()
    pred = model(val_input)
    return criterion(pred, val_target)
    
def accuracy(model, inputs, targets):
    preds = model(inputs)
    return (preds.argmax(axis=1) == targets).long().sum().item() / targets.shape[0]

def train_all(train_input, train_target, train_classes, val_input, val_target, val_classes,test_input, test_target, test_classes, niter = 15, nb_epochs = 25,mini_batch_size = 100):
    accuracies_train = []
    accuracies_test = []
    accuracies_val = []
    losses = np.zeros((niter,nb_epochs))
    losses_val = np.zeros((niter,nb_epochs))
    # flatten
    train_input = train_input.view(train_input.size(0), -1)
    test_input = test_input.view(test_input.size(0), -1)
    val_input = val_input.view(val_input.size(0), -1)
    for i in range(niter):
        print("-"*50,f" \n Iteration {i} \n ")   
        # define the model
        model = MLP() 
        # train model
        losses[i,:],losses_val[i,:] = train(model, train_input, train_target,val_input, val_target, mini_batch_size,nb_epochs=nb_epochs)
        print(f"Baseline Training accuracy is {accuracy(model,train_input,train_target)} ")
        test_accuracy = accuracy(model,test_input,test_target)
        train_accuracy = accuracy(model,train_input,train_target)
        val_accuracy = accuracy(model,val_input,val_target)
        accuracies_train.append(train_accuracy )
        accuracies_test.append(test_accuracy )
        accuracies_val.append(val_accuracy )
        print(f"Baseline Validation accuracy is {val_accuracy} ")
    
    return losses, losses_val, accuracies_train, accuracies_test, accuracies_val
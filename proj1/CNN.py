import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

class CNN(nn.Module):
    """ Modified LeNet. Takes input format 1 x 14 x 14 and outputs if the first digit is smaller than the first"""
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120,84),
            nn.ReLU(inplace=True),
            nn.Linear(84,2)
)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x 
    
def train(model, train_input, train_target, mini_batch_size, verbose = False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for e in range(25):
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
        

def accuracy(preds, targets):
    return (preds.argmax(axis=1) == targets.argmax(axis=1)).long().sum().item() / targets.shape[0]

def get_mis_class(model,input_,target,classes):
    preds = model(input_).argmax(axis=1) == target
    misclassified = classes[~preds]
    return misclassified.tolist()
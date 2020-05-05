import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np
import matplotlib.pyplot as plt


class LeNetLike(nn.Module):
    """ Modified LeNet. Takes input format 1 x 14 x 14 and outputs the class"""
    def __init__(self):
        super(LeNetLike, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size = 3),
            nn.ReLU()
)
        self.classifier = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),
            nn.Softmax(dim=1)
)
    def forward(self, input_):
        output = []
        for i in range(2):
            x = input_[:,i,...].unsqueeze(1)
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            output.append(x)
        return output
    
######################################################################################

def aux_loss(output,classes,target,criterion):
    # loss of prediction of the two number
    loss_1 = criterion(output[0], classes[:,0])
    loss_2 = criterion(output[1],  classes[:,1])
    
    # loss of comparaison between the two. 
    val_1 = (output[0] @ torch.arange(10).float()).unsqueeze(1)
    val_2 = (output[1] @ torch.arange(10).float()).unsqueeze(1)
    vals = torch.cat((val_2,val_1),1)
    # normalize it 
    vals = vals/torch.cat(2*[vals.sum(1).unsqueeze(1)],1)
    loss_3 = criterion(vals,  target.long())
    
    return loss_1 + loss_2 + loss_3

######################################################################################

def train(model, train_input, train_classes, train_target, mini_batch_size, verbose = False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    for e in range(25):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            optimizer.zero_grad()
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = aux_loss(output,train_classes.narrow(0, b, mini_batch_size),train_target.narrow(0, b, mini_batch_size),criterion)
            sum_loss = sum_loss + loss.item()
            loss.backward()
            optimizer.step()
        if verbose:
            print(e,sum_loss)
    return sum_loss
            
    
######################################################################################


# Accuracy for the baseline trains to get the classes then compare predicted class
def accuracy(model,inputs,targets):
    """ INPUT : 
        - model: model that predicts the digit values 
        - input_long: the input of the format 2000 x 10
        - tagets: ground truth of the pairs comparaison with format 1000 """
    # Predict class of inputs
    output = model(inputs)
    preds = output[0].argmax(dim=1) <= output[1].argmax(dim=1)
    # Compute accuracy
    accuracy = (preds == targets).long().sum().item()/len(preds)
    return accuracy 

######################################################################################

def get_mis_class(model,input_,target,classes):
    output = model(input_)
    preds = output[0].argmax() <= output[1].argmax()
    misclassified = classes[~preds]
    return misclassified.tolist()

######################################################################################
def plotloss(losses):
    mean = losses.mean(axis=1)
    error = np.sqrt(losses.var(axis=1))
    x = np.linspace(0, len(mean), len(mean))

    plt.plot(x, mean, 'k-')
    plt.fill_between(x, mean-error, mean+error)
    plt.show()
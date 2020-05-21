import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np

class CNNr(nn.Module):
    """ Modified LeNet. Takes input format 2 x 14 x 14 and outputs if the first digit is smaller than the first"""
    def __init__(self):
        super(CNNr, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(32, 64, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3)
)
        self.classifier = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(120,84),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(84,2)
)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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

def accuracy(preds, targets):
    return (preds.argmax(axis=1) == targets).long().sum().item() / targets.shape[0]

def get_mis_class(model,input_,target,classes):
    preds = model(input_).argmax(axis=1) == target
    misclassified = classes[~preds]
    return misclassified.tolist()

def train_all(train_input, train_target, train_classes, val_input, val_target, val_classes,test_input, test_target, test_classes, niter = 15, nb_epochs = 25,mini_batch_size = 100):

    all_classified = []
    misclassified = []
    accuracies_train = []
    accuracies_test = []
    accuracies_val = []
    losses = np.zeros((niter, nb_epochs))
    losses_val = np.zeros((niter, nb_epochs))
    for i in range(niter):
        print("-"*50,f" \n Iteration {i} \n ")

        # define the model
        model = CNNr() 
        # train model
        losses[i,:],losses_val[i,:] = train(model, train_input, train_target,val_input, val_target, mini_batch_size,nb_epochs=nb_epochs)
        model = model.eval()
        train_accuracy = accuracy(model(train_input),train_target)
        test_accuracy = accuracy(model(test_input),test_target)
        val_accuracy = accuracy(model(val_input),val_target)

        misclass = get_mis_class(model,  torch.cat((test_input,val_input)), torch.cat((test_target,val_target)),
                                 torch.cat((test_classes,val_classes)))
        [all_classified.append(x) for x in torch.cat((test_classes,val_classes))]
        [misclassified.append(x) for x in misclass ]
        accuracies_train.append(train_accuracy )
        accuracies_test.append(test_accuracy )
        accuracies_val.append(val_accuracy )

        print(f"Training accuracy is {train_accuracy} ")
        print(f"Validation accuracy is {val_accuracy} ")
    return losses, losses_val, accuracies_train, accuracies_test, accuracies_val,all_classified,misclassified
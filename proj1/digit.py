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
            nn.ReLU(),
)
        self.classifier = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 62),
            nn.ReLU(inplace=True),
            nn.Linear(62, 10)
)
    def forward(self, inputs):
        x = self.features(inputs)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
######################################################################################

def aux_loss(output_1,output_2,classes,criterion):
    # loss of prediction of the two number
    loss_1 = criterion(output_1, classes[:,0])
    loss_2 = criterion(output_2,  classes[:,1])
    
    return loss_1, loss_2

######################################################################################

def train(model, train_input, train_classes,val_input, val_classes, mini_batch_size,nb_epochs=25, verbose = False):
    losses = np.zeros(nb_epochs)
    val_losses = np.zeros(nb_epochs)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    for e in range(nb_epochs):
        model = model.train()
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            optimizer.zero_grad()
            output_1 = model(train_input.narrow(0, b, mini_batch_size)[:,0].unsqueeze(1))
            output_2 = model(train_input.narrow(0, b, mini_batch_size)[:,1].unsqueeze(1))
            loss_1,loss_2  = aux_loss(output_1,output_2,train_classes.narrow(0, b, mini_batch_size),criterion)
            loss = loss_1 + loss_1
            sum_loss = sum_loss + loss.item()
            loss.backward()
            optimizer.step()
        if verbose:
            print(e,sum_loss)
        losses[e] = sum_loss
        val_losses[e] = get_loss_val(model.eval(), val_input, val_classes)
    return losses,val_losses
            
    
######################################################################################


# Accuracy for the baseline trains to get the classes then compare predicted class
def accuracy(model,inputs,targets):
    """ INPUT : 
        - model: model that predicts the digit values 
        - input_long: the input of the format 2000 x 10
        - tagets: ground truth of the pairs comparaison with format 1000 """
    # Predict class of inputs
    output_1 = model(inputs[:,0].unsqueeze(1))
    output_2 = model(inputs[:,1].unsqueeze(1))
    preds = output_1.argmax(dim=1) <= output_2.argmax(dim=1)
    # Compute accuracy
    accuracy = (preds == targets).long().sum().item()/len(preds)
    return accuracy 

######################################################################################

def get_mis_class(model,inputs,target,classes):
    output_1 = model(inputs[:,0].unsqueeze(1))
    output_2 = model(inputs[:,1].unsqueeze(1))
    preds = output_1.argmax(dim=1) <= output_2.argmax(dim=1)
    mis = preds == target
    misclassified = classes[~mis]
    return misclassified.tolist(),mis

######################################################################################

def get_loss_val(model, val_input,val_classes):
    criterion = nn.CrossEntropyLoss()
    output_1 = model(val_input[:,0].unsqueeze(1))
    output_2 = model(val_input[:,1].unsqueeze(1))
    classes_1 = val_classes[:,0]
    classes_2 = val_classes[:,1]
    return criterion(output_1, classes_1) + criterion(output_2, classes_2)

######################################################################################

def train_all(train_input, train_target, train_classes, val_input, val_target, val_classes,test_input, test_target, test_classes, niter = 15, nb_epochs = 25,mini_batch_size = 100):

    all_classified = []
    misclassified = []
    accuracies_train = []
    accuracies_test = []
    accuracies_val = []
    
    losses = np.zeros((niter,nb_epochs))
    losses_val = np.zeros((niter,nb_epochs))

    for i in range(niter):
        print("-"*50,f" \n Iteration {i} \n ")

        # define the model
        model =  LeNetLike() 

        # train model

        losses[i, :], losses_val[i, :] = train(model, train_input, train_classes,
                                               val_input, val_classes, mini_batch_size, nb_epochs=nb_epochs)
        train_accuracy = accuracy(model,train_input,train_target)
        test_accuracy = accuracy(model,test_input,test_target)
        val_accuracy = accuracy(model,val_input,val_target)

        misclass,misbool = get_mis_class(model,  torch.cat((test_input,val_input)), torch.cat((test_target,val_target)),
                                 torch.cat((test_classes,val_classes)))
        [all_classified.append(x) for x in torch.cat((test_classes,val_classes))]
        [misclassified.append(x) for x in misclass ]
        accuracies_train.append(train_accuracy )
        accuracies_test.append(test_accuracy )
        accuracies_val.append(val_accuracy )
        print(f"Training accuracy is {train_accuracy} ")
        print(f"Validation accuracy is {val_accuracy} ")
    
    return losses, losses_val, accuracies_train, accuracies_test, accuracies_val,all_classified,misclassified,misbool 
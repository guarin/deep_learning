import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np

class CNNp(nn.Module):
    """ Modified LeNet. Takes input format 1 x 14 x 14 and outputs the class"""
    def __init__(self):
        super(CNNp, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(),
            nn.Conv2d(6, 16, kernel_size = 3),
            nn.ReLU(),
             nn.Dropout2d()
)
        self.classifier = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(84, 10),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(10, 1)
)
    def forward(self, input_):
        output = []
        for i in range(2):
            x = input_[:,i,...].unsqueeze(1)
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            output.append(x)
        output = torch.sigmoid(output[1] - output[0]).squeeze()
        return output
    
######################################################################################


# Accuracy for the baseline trains to get the classes then compare predicted class
def accuracy(model,inputs,targets):
    """ INPUT : 
        - model: model that predicts the digit values 
        - input_long: the input of the format 2000 x 10
        - tagets: ground truth of the pairs comparaison with format 1000 """

    # Predict class of inputs
    preds = model(inputs).round()
    # Compute accuracy
    accuracy = (preds == targets).long().sum().item()/len(preds)
    return accuracy 

######################################################################################

def train(model, train_input, train_target, mini_batch_size, verbose = False):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for e in range(25):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            optimizer.zero_grad()
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size).float())
            sum_loss = sum_loss + loss.item()
            loss.backward()
            optimizer.step()
        if verbose:
            print(e,sum_loss)
    return sum_loss

######################################################################################

def get_loss_val(model, val_input,val_target):
    criterion = nn.MSELoss()
    pred = model(val_input)
    return criterion(pred, val_target)

######################################################################################
            
def get_mis_class(model,input_,target,classes):
    preds = model(input_).round() == target
    misclassified = classes[~preds]
    return misclassified.tolist()

def train_all(train_input, train_target, train_classes, val_input, val_target, val_classes,test_input, test_target, test_classes, niter = 15, nround = 25,mini_batch_size = 100):

    all_classified = []
    misclassified = []
    accuracies_train = []
    accuracies_test = []
    accuracies_val = []
    losses = np.zeros((niter,nround))
    losses_val = np.zeros((niter,nround))

    for i in range(niter):
        print("-"*50,f" \n Iteration {i} \n ")

        # define the model
        model =  CNNp() 

        # train model
        for k in range(nround):
            losses[i,k] = train(model.train(), train_input, train_target, mini_batch_size)
            losses_val[i,k] = get_loss_val(model.eval(), val_input,val_target)
        train_accuracy = accuracy(model,train_input,train_target)
        test_accuracy = accuracy(model,test_input,test_target)
        val_accuracy = accuracy(model,val_input,val_target)

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
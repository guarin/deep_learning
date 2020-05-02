import torch
from torch import nn
from torch.nn import functional as F


# Baseline model is a variation of LeNet that simply tries to classify the digits. 

class LeNetLike(nn.Module):
    """ Modified LeNet. Takes input format 1 x 14 x 14 and outputs 10 classes one-hot encoded"""
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
            nn.Linear(84, 10)
)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
######################################################################################


# Accuracy for the baseline trains to get the classes then compare predicted class
def accuracy(model,input_long,targets):
    """ INPUT : 
        - model: model that predicts 10 digit class using one-hot 
        - input_long: the input of the format 2000 x 10
        - tagets: ground truth of the pairs comparaison with format 1000 """
    # Predict class of inputs
    preds_long = model(input_long)
    # Compare the pairs
    preds = (preds_long[:1000,:].argmax(dim=1) <= preds_long[1000:,:].argmax(dim=1)).long()
    # Compute accuracy
    accuracy = (preds == targets).long().sum().item()/len(preds)
    return accuracy 
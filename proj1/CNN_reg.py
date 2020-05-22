import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

class CNNr(nn.Module):
    """ Modified LeNet. Takes input format 2 x 14 x 14 and outputs if the first digit is smaller than the first"""
    def __init__(self):
        super(CNNr, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size = 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(32, 64, kernel_size = 3),
            nn.ReLU(inplace=True),
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


# Training function
# -----------------------------------------------------------------------------------
def train(model, train_input, train_target, train_classes, mini_batch_size, nb_epochs=25, verbose = False):
    """
    Train the model
    Params:
    model 	        : defined network
    train_input     : train input data
    train_target    : train target data
    train_classes   : train digit classes
    minibatch_size  : size of each minibatch
    np_epochs       : number of epochs to train the model (default 25)
    verbose         : verbosity of training routine
    Returns:
    None, the model is trained inplace.
    """
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


# Accuracy function
# -----------------------------------------------------------------------------------
def accuracy(model, inputs, targets):
    """ INPUT:
        - model: model that predicts the digit values
        - inputs: the input to be predicted
        - targets: ground truth of the pairs comparison
        OUTPUT:
        - accuracy: Percentage score of correct predictions
    """
    model.eval()
    return (model(inputs).argmax(axis=1) == targets).long().sum().item() / targets.shape[0]


def get_model():
    return CNNr()
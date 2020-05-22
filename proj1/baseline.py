import torch
from torch import nn
from torch.nn import functional as F
from torch import optim


# Model class definition
# -----------------------------------------------------------------------------------
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
    # flatten
    train_input = train_input.view(train_input.size(0), -1)

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
    # flatten
    inputs = inputs.view(inputs.size(0), -1)
    model.eval()
    preds = model(inputs)
    return (preds.argmax(axis=1) == targets).long().sum().item() / targets.shape[0]


# Get Model function
# -----------------------------------------------------------------------------------
def get_model():
    return MLP()
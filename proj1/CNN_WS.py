import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

class CNNrws(nn.Module):
    # siamese CNN with shared weights and dropout regularization
    def __init__(self):
        super(CNNrws, self).__init__()
        #Layers image
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)

        #Combining layers
        self.lin4 = nn.Linear(168,20)
        self.lin5 = nn.Linear(20,2)

    def forward(self, x):
        y = []
        for i in range(2):
            h1 = F.relu(F.max_pool2d(self.conv1(x[:,i,:,:].view(x.size(0),1,14,14)), kernel_size=2), inplace=True)
            h1 = F.dropout2d(h1, 0.3)
            h2 = F.relu(F.max_pool2d(self.conv2(h1), kernel_size=2, stride=2), inplace=True)
            h2 = F.dropout2d(h2, 0.3)
            h3 = F.relu(self.fc1(h2.view((-1, 256))), inplace=True)
            h3 = F.dropout(h3, 0.3)
            h4 = F.relu(self.fc2(h3), inplace=True)
            h4 = F.dropout(h4, 0.3)

            y.append(h4)

        y1 = F.relu(self.lin4(torch.cat((y[0].view(-1, 84), y[1].view(-1, 84)), 1)), inplace=True)
        y2 = self.lin5(y1)
        return y2


# Training function
# -----------------------------------------------------------------------------------
def train(model, train_input, train_target, train_classes, mini_batch_size, nb_epochs=25, verbose=False):
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
        model = model.train()
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            optimizer.zero_grad()
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            loss.backward()
            optimizer.step()
        if verbose:
            print(e, sum_loss)


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
    return CNNrws()
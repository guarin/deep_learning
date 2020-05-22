import torch
from torch import nn
from torch.nn import functional as F
from torch import optim


class CNN_aux(nn.Module):
    # siamese CNN with shared weights and dropout regularization
    def __init__(self):
        super(CNN_aux, self).__init__()
        #Layers image 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        #Combining layers
        self.lin4 = nn.Linear(168,20)
        self.lin5 = nn.Linear(20,2)

    def forward(self, x):
        y, z = [], []
        for i in range(2):
            h1 = F.relu(F.max_pool2d(self.conv1(x[:,i,:,:].view(x.size(0),1,14,14)), kernel_size=2), inplace=True)
            h1 = F.dropout2d(h1, 0.3)
            h2 = F.relu(F.max_pool2d(self.conv2(h1), kernel_size=2, stride=2), inplace=True)
            h2 = F.dropout2d(h2, 0.3)
            h3 = F.relu(self.fc1(h2.view((-1, 256))), inplace=True)
            h3 = F.dropout(h3, 0.3)
            h4 = F.relu(self.fc2(h3), inplace=True)
            h4 = F.dropout(h4, 0.3)
            z.append(h4)
            h5 = self.fc3(h4)
            y.append(h5)

        # Classifiction
        y1 = F.relu(self.lin4(torch.cat((z[0].view(-1, 84), z[1].view(-1, 84)), 1)), inplace=True)
        y2 = self.lin5(y1)
        return [y2, y[0], y[1]]


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
        model = model.train()
        beta = (nb_epochs - e) / nb_epochs
        sum_loss = 0
        sum_aux_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output[0], train_target.narrow(0, b, mini_batch_size))
            aux_loss = criterion(output[1], train_classes[:, 0].narrow(0, b, mini_batch_size)) + \
                       criterion(output[2], train_classes[:, 1].narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            sum_aux_loss = sum_aux_loss + aux_loss.item()
            model.zero_grad()
            ((1 - beta) * loss + beta * aux_loss).backward()
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
    return (model(inputs)[0].argmax(axis=1) == targets).long().sum().item() / targets.shape[0]


def get_model():
    return CNN_aux()
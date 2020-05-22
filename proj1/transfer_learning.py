import torch
from torch import nn
from torch.nn import functional as F
from torch import optim


class BaseNet(nn.Module):
    # First network only for number classification task
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x.view(-1,1,14,14)), kernel_size=2, stride=2))
        x = F.dropout2d(x, 0.3)
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.dropout2d(x, 0.3)
        x = F.relu(self.fc1(x.view((-1, 256))))
        x = F.dropout(x, 0.3)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.3)
        x = self.fc3(x)
        return x

class Full_Net(nn.Module):
    # Second network to be initialized with Base Network, to correctly predict binary target
    def __init__(self):
        super(Full_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # Combining layers
        self.lin4 = nn.Linear(168, 20)
        self.lin5 = nn.Linear(20, 2)

    def forward(self, x):
        y = []
        # treat images separately
        for i in range(2):
            h1 = F.relu(F.max_pool2d(self.conv1(x[:,i,:,:].view(x.size(0),1,14,14)), kernel_size=2), inplace=True)
            h1 = F.dropout2d(h1, 0.3)
            h2 = F.relu(F.max_pool2d(self.conv2(h1), kernel_size=2, stride=2), inplace=True)
            h2 = F.dropout2d(h2, 0.3)
            h3 = F.relu(self.fc1(h2.view((-1, 256))), inplace=True)
            h3 = F.dropout(h3, 0.3)
            h4 = F.relu(self.fc2(h3), inplace=True)
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
    model1, model2 = model
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=1e-3)

    train_input_ = torch.cat((train_input[:, 0, :, :], train_input[:, 1, :, :]))
    train_target_ = torch.cat((train_classes[:, 0], train_classes[:, 1]))

    for e in range(nb_epochs):
        sum_loss = 0
        for b in range(0, train_input_.size(0), mini_batch_size):
            output = model1(train_input_.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target_.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            model1.zero_grad()
            loss.backward()
            optimizer1.step()
        if verbose:
            print(e, sum_loss)

    model2.load_state_dict(model1.state_dict(), strict=False)
    for p in model2.conv1.parameters():
        p.requires_grad = False
    for p in model2.conv2.parameters():
        p.requires_grad = False

    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)
    for e in range(nb_epochs):
        model2 = model2.train()
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model2(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            model2.zero_grad()
            loss.backward()
            optimizer2.step()
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
    model[0].eval()
    model[1].eval()
    return (model[1](inputs).argmax(axis=1) == targets).long().sum().item() / targets.shape[0]


# Get Model function
# -----------------------------------------------------------------------------------
def get_model():
    return BaseNet(), Full_Net()
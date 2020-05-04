import torch
from torch import nn
from torch.nn import functional as F
from torch import optim


class MLP_num_pred(nn.Module):
    # Basic Model to predict mnist numbers with only FCs.
    def __init__(self):
        super(MLP_num_pred, self).__init__()
        self.lin1a = nn.Linear(196, 784)
        self.lin2a = nn.Linear(784, 32)
        self.lin3a = nn.Linear(32, 10)

    def forward(self, x):
        h1a = F.relu(self.lin1a(x))
        h2a = F.relu(self.lin2a(h1a))
        h3a = F.relu((self.lin3a(h2a)))
        return h3a

class MLP_aux_loss(nn.Module):
    # Basic Model from above used to introduce auxiliary loss in classification task
    def __init__(self):
        super(MLP_aux_loss, self).__init__()
        #Layers image 1
        self.lin1a = nn.Linear(196, 784)
        self.lin2a = nn.Linear(784, 32)
        self.lin3a = nn.Linear(32, 10)

        #Layers image 2
        self.lin1b = nn.Linear(196, 784)
        self.lin2b = nn.Linear(784, 32)
        self.lin3b = nn.Linear(32, 10)

        #Combining layers
        self.lin4 = nn.Linear(64,10)
        self.lin5 = nn.Linear(10,2)

    def forward(self, x):
        h1a = F.relu(self.lin1a(x[:,0,:]))
        h2a = F.relu(self.lin2a(h1a))
        h3a = F.relu((self.lin3a(h2a)))

        # Image 2 class
        h1b = F.relu(self.lin1b(x[:,1,:]))
        h2b = F.relu(self.lin2b(h1b))
        h3b = F.relu(self.lin3b(h2b))

        # Classifiction
        y1 = F.relu(self.lin4(torch.cat((h2a.view(-1, 32), h2b.view(-1, 32)), 1)))
        y2 = F.relu(self.lin5(y1))
        return [y2, h3a, h3b]

def train_MLP_num_pred(model, train_input, train_target, nb_epochs, mini_batch_size, lr = 0.1, verbose=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr)
    for e in range(nb_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose:
            print(e, sum_loss)

def train_MLP_aux(model, train_input, train_target, train_class, nb_epochs, mini_batch_size, lr = 0.1, aux_loss = True, verbose=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr)
    for e in range(nb_epochs):
        sum_loss = 0
        if aux_loss:
            beta = (nb_epochs - e) / nb_epochs
        else:
            beta = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output[0], train_target.narrow(0, b, mini_batch_size))
            aux_loss = criterion(output[1], train_class[:,0].narrow(0, b, mini_batch_size)) + \
                        criterion(output[2], train_class[:,1].narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            model.zero_grad()
            ((1 - beta) * loss + beta * aux_loss).backward()
            optimizer.step()
        if verbose:
            print(e, sum_loss)

def accuracy(model, inputs, targets):
    # for 2 classes classification tastk, number of wrong classifications/samples
    preds = torch.argmax(model(inputs)[0], dim = 1)
    return (preds == targets).long().sum().item() / targets.shape[0]

def accuracy_num_pred(model, input, target, mini_batch_size):
    # For 10 classes classification task, number of wrong classifications/samples
    output = model(input)
    _, preds = output.max(1)
    return (preds == target).long().sum().item()/target.shape[0]
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import proj1.projet1_helpers as helpers

class BaseNet(nn.Module):
    # First network only for number classification task
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x.view(-1,1,14,14)), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x

class Full_Net(nn.Module):
    # Second network to be initialized with Base Network, to correctly predict binary target
    def __init__(self):
        super(Full_Net, self).__init__()
        #Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 10)
        #Combining layers
        self.lin4 = nn.Linear(240,20)
        self.lin5 = nn.Linear(20,2)

    def forward(self, x):
        # Image 1 class
        h1a = F.relu(F.max_pool2d(self.conv1(x[:,0,:,:].view(x.size(0),1,14,14)), kernel_size=2))
        h2a = F.relu(F.max_pool2d(self.conv2(h1a), kernel_size=2, stride=2))
        h3a = F.relu((self.fc1(h2a.view((-1, 256)))))
        h4a = self.fc2(h3a)
        # Image 2 class
        h1b = F.relu(F.max_pool2d(self.conv1(x[:,1,:,:].view(x.size(0),1,14,14)), kernel_size=2))
        h2b = F.relu(F.max_pool2d(self.conv2(h1b), kernel_size=2, stride=2))
        h3b = F.relu((self.fc1(h2b.view(-1,256))))
        h4b = self.fc2(h3b)
        # Classifiction
        y1 = F.relu(self.lin4(torch.cat((h3a.view(-1, 120), h3b.view(-1, 120)), 1)))
        y2 = self.lin5(y1)
        return [y2, h4a, h4b]

def train(model1, model2, train_input, train_target, train_class, nb_epochs, mini_batch_size, lr=0.1, verbose=False):
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(model1.parameters(), lr)

    train_input_ = torch.cat((train_input[:, 0, :, :], train_input[:, 1, :, :]))
    train_target_ = torch.cat((train_class[:, 0], train_class[:, 1]))

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

    optimizer2 = optim.SGD(model2.parameters(), lr)
    for e in range(nb_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model2(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output[0], train_target.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            model2.zero_grad()
            loss.backward()
            optimizer2.step()
        if verbose:
            print(e, sum_loss)

def accuracy(model, inputs, targets):
    # for 2 classes classification tasks, number of wrong classifications/samples
    preds = torch.argmax(model(inputs)[0], dim = 1)
    return (preds == targets).long().sum().item() / targets.shape[0]

def eval(runs, mb_size=100, lr=0.1, epochs=25, aux_loss=True, verbose=False):
    accuracies = []
    for i in range(runs):
        if verbose:
            print("-" * 50, f" \n Iteration {i} \n ")
        # Generate the pairs
        train_input, train_target, train_classes, test_input, test_target, test_classes = helpers.load_data(1000)

        # define the model
        model1 = BaseNet()
        model2 = Full_Net()

        # train model
        train(model1, model2, train_input, train_target, train_classes, epochs, mb_size, lr, verbose)
        if verbose:
            print(f"Baseline Training accuracy is {accuracy(model2, train_input, train_target)} ")
        test_accuracy = accuracy(model2, test_input, test_target)
        accuracies.append(test_accuracy)
        if verbose:
            print(f"Baseline Test accuracy is {test_accuracy} ")

    accs = torch.Tensor(accuracies)
    print(f"The accuracy of the model is {accs.mean():.4f} ± {accs.var():.4f} ")

    helpers.plot_performance(accuracies, runs)
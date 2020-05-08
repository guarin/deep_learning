import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import proj1.projet1_helpers as helpers

class LeNet_aux_loss(nn.Module):
    # CNN with auxiliary loss only, separate weights for first and second image
    def __init__(self):
        super(LeNet_aux_loss, self).__init__()
        #Layers image 1
        self.conv1a = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1a = nn.Linear(256, 120)
        self.fc2a = nn.Linear(120, 84)
        self.fc3a = nn.Linear(84, 10)

        #Layers image 2
        self.conv1b = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2b = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1b = nn.Linear(256, 120)
        self.fc2b = nn.Linear(120, 84)
        self.fc3b = nn.Linear(84, 10)

        #Combining layers
        self.lin4 = nn.Linear(240,20)
        self.lin5 = nn.Linear(20,2)

    def forward(self, x):
        # Image 1 class
        h1a = F.relu(F.max_pool2d(self.conv1a(x[:,0,:,:].view(x.size(0),1,14,14)), kernel_size=2))
        h2a = F.relu(F.max_pool2d(self.conv2a(h1a), kernel_size=2, stride=2))
        h3a = F.relu((self.fc1a(h2a.view((-1, 256)))))
        h4a = F.relu((self.fc2a(h3a)))
        h5a = self.fc3a(h4a)

        # Image 2 class
        h1b = F.relu(F.max_pool2d(self.conv1b(x[:,1,:,:].view(x.size(0),1,14,14)), kernel_size=2))
        h2b = F.relu(F.max_pool2d(self.conv2b(h1b), kernel_size=2, stride=2))
        h3b = F.relu((self.fc1b(h2b.view(-1,256))))
        h4b = F.relu(self.fc2b(h3b))
        h5b = self.fc3b(h4b)

        # Classifiction
        y1 = F.relu(self.lin4(torch.cat((h3a.view(-1, 120), h3b.view(-1, 120)), 1)))
        y2 = self.lin5(y1)
        return [y2, h5a, h5b]

def train(model, train_input, train_target, train_class, nb_epochs, mini_batch_size, lr=0.1, aux_loss=True, verbose=False):
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
            aux_loss = criterion(output[1], train_class[:, 0].narrow(0, b, mini_batch_size)) + \
                       criterion(output[2], train_class[:, 1].narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            model.zero_grad()
            ((1 - beta) * loss + beta * aux_loss).backward()
            optimizer.step()
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
        model = LeNet_aux_loss()

        # train model
        train(model, train_input, train_target, train_classes, epochs, mb_size, lr, aux_loss, verbose)
        if verbose:
            print(f"Baseline Training accuracy is {accuracy(model, train_input, train_target)} ")
        test_accuracy = accuracy(model, test_input, test_target)
        accuracies.append(test_accuracy)
        if verbose:
            print(f"Baseline Test accuracy is {test_accuracy} ")

    accs = torch.Tensor(accuracies)
    print(f"The accuracy of the model is {accs.mean():.4f} ± {accs.var():.4f} ")

    helpers.plot_performance(accuracies, runs)
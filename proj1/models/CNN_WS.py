import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import proj1.projet1_helpers as helpers

class CNNrws(nn.Module):
    # siamese CNN with shared weights and dropout regularization
    def __init__(self):
        super(CNNrws, self).__init__()
        #Layers image 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)

        #Combining layers
        self.lin4 = nn.Linear(168,20)
        self.lin5 = nn.Linear(20,2)

    def forward(self, x):
        # Image 1 class
        h1a = F.relu(F.max_pool2d(self.conv1(x[:,0,:,:].view(x.size(0),1,14,14)), kernel_size=2))
        h1a = F.dropout2d(h1a, 0.7)
        h2a = F.relu(F.max_pool2d(self.conv2(h1a), kernel_size=2, stride=2))
        #h2a = F.dropout2d(h2a, 0.7)
        h3a = F.relu(self.fc1(h2a.view((-1, 256))))
        #h3a = F.dropout(h3a, 0.7)
        h4a = F.relu(self.fc2(h3a))
        #h4a = F.dropout(h4a, 0.7)


        # Image 2 class
        h1b = F.relu(F.max_pool2d(self.conv1(x[:,1,:,:].view(x.size(0),1,14,14)), kernel_size=2))
        h1b = F.dropout2d(h1b, 0.7)
        h2b = F.relu(F.max_pool2d(self.conv2(h1b), kernel_size=2, stride=2))
        #h2b = F.dropout2d(h2b, 0.7)
        h3b = F.relu((self.fc1(h2b.view(-1,256))))
        #h3b = F.dropout(h3b, 0.7)
        h4b = F.relu(self.fc2(h3b))
        #h4b = F.dropout(h4b, 0.7)


        # Classifiction
        y1 = F.relu(self.lin4(torch.cat((h4a.view(-1, 84), h4b.view(-1, 84)), 1)))
        y2 = self.lin5(y1)
        return [y2]

def train(model, train_input, train_target, train_class, nb_epochs, mini_batch_size, lr=0.001, verbose=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    for e in range(nb_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output[0], train_target.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose:
            print(e, sum_loss)

def accuracy(model, inputs, targets):
    # for 2 classes classification tasks, number of wrong classifications/samples
    preds = torch.argmax(model(inputs)[0], dim = 1)
    return (preds == targets).long().sum().item() / targets.shape[0]

def eval(runs, mb_size=100, lr=0.001, epochs=25, aux_loss=True, verbose=False):
    accuracies = []
    for i in range(runs):
        if verbose:
            print("-" * 50, f" \n Iteration {i} \n ")
        # Generate the pairs
        train_input, train_target, train_classes, test_input, test_target, test_classes = helpers.load_data(1000)

        # define the model
        model = CNNrws()

        # train model
        train(model, train_input, train_target, train_classes, epochs, mb_size, lr, verbose)
        if verbose:
            print(f"Baseline Training accuracy is {accuracy(model, train_input, train_target)} ")
        test_accuracy = accuracy(model, test_input, test_target)
        accuracies.append(test_accuracy)
        if verbose:
            print(f"Baseline Test accuracy is {test_accuracy} ")

    accs = torch.Tensor(accuracies)
    print(f"The accuracy of the model is {accs.mean():.4f} ± {accs.var():.4f} ")

    helpers.plot_performance(accuracies, runs)
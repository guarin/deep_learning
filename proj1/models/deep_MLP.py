import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import proj1.projet1_helpers as helpers

class MLP_aux_loss(nn.Module):
    # Deep model with auxiliary loss
    def __init__(self):
        super(MLP_aux_loss, self).__init__()
        #Layers image 1
        self.lin1a = nn.Linear(196, 368)
        self.lin2a = nn.Linear(368, 472)
        self.lin3a = nn.Linear(472, 594)
        self.lin4a = nn.Linear(594, 636)
        self.lin5a = nn.Linear(636, 784)
        self.lin6a = nn.Linear(784, 32)
        self.lin7a = nn.Linear(32, 10)

        #Layers image 2
        self.lin1b = nn.Linear(196, 368)
        self.lin2b = nn.Linear(368, 472)
        self.lin3b = nn.Linear(472, 594)
        self.lin4b = nn.Linear(594, 636)
        self.lin5b = nn.Linear(636, 784)
        self.lin6b = nn.Linear(784, 32)
        self.lin7b = nn.Linear(32, 10)

        #Combining layers
        self.lin8 = nn.Linear(64,10)
        self.lin9 = nn.Linear(10,2)

    def forward(self, x):
        # Image 1 class
        h1a = F.relu(self.lin1a(x[:,0,:]))
        h2a = F.relu(self.lin2a(h1a))
        h3a = F.relu(self.lin3a(h2a))
        h4a = F.relu(self.lin4a(h3a))
        h5a = F.relu(self.lin5a(h4a))
        h6a = F.relu(self.lin6a(h5a))
        h8a = self.lin7a(h6a)

        # Image 2 class
        h1b = F.relu(self.lin1b(x[:,1,:]))
        h2b = F.relu(self.lin2b(h1b))
        h3b = F.relu(self.lin3b(h2b))
        h4b = F.relu(self.lin4b(h3b))
        h5b = F.relu(self.lin5b(h4b))
        h6b = F.relu(self.lin6b(h5b))
        h8b = self.lin7b(h6b)

        # Classifiction
        y1 = F.relu(self.lin8(torch.cat((h6a.view(-1, 32), h6b.view(-1, 32)), 1)))
        y2 = self.lin9(y1)
        return [y2, h8a, h8b]

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

        # flatten
        train_input = train_input.view(train_input.size(0), 2, -1)
        test_input = test_input.view(test_input.size(0), 2, -1)

        # define the model
        model = MLP_aux_loss()

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
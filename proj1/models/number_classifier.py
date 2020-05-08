import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import proj1.projet1_helpers as helpers

class BaseNet(nn.Module):
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

def train(model, train_input, train_target, nb_epochs, mini_batch_size, lr=0.1, verbose=False):
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

def accuracy(model, inputs, targets):
    # for 10 classes classification tasks, number of wrong classifications/samples
    preds = torch.argmax(model(inputs), dim = 1)
    return (preds == targets).long().sum().item() / targets.shape[0]

def eval(runs, mb_size=100, lr=0.1, epochs=25, verbose=False):
    accuracies = []
    for i in range(runs):
        if verbose:
            print("-" * 50, f" \n Iteration {i} \n ")
        # Generate the numbers
        train_input, train_target, train_classes, test_input, test_target, test_classes = helpers.load_data(1000)
        train_input_ = torch.cat((train_input[:, 0, :, :], train_input[:, 1, :, :]))
        train_target_ = torch.cat((train_classes[:, 0], train_classes[:, 1]))
        test_input_ = torch.cat((test_input[:, 0, :, :], test_input[:, 1, :, :]))
        test_target_ = torch.cat((test_classes[:, 0], test_classes[:, 1]))

        # define the model
        model = BaseNet()

        # train model
        train(model, train_input_, train_target_, epochs, mb_size, lr, verbose)
        if verbose:
            print(f"Baseline Training accuracy is {accuracy(model, train_input_, train_target_)} ")

        preds = torch.argmax(model(test_input_), dim = 1)

        pred1 = preds[:int(test_target_.shape[0]/2)]
        pred2 = preds[int(test_target_.shape[0]/2):]

        test_accuracy = ((pred1 <= pred2).long() == test_target).long().sum().item() / test_target.shape[0]

        #test_accuracy = accuracy(model, test_input_, test_target_)
        accuracies.append(test_accuracy)
        if verbose:
            print(f"Baseline Test accuracy is {test_accuracy} ")

    accs = torch.Tensor(accuracies)
    print(f"The accuracy of the model is {accs.mean():.4f} ± {accs.var():.4f} ")

    helpers.plot_performance(accuracies, runs)
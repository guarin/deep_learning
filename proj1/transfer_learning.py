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
        # Layers
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
        for i in range(2):
            h1 = F.relu(F.max_pool2d(self.conv1(x[:,i,:,:].view(x.size(0),1,14,14)), kernel_size=2), inplace=True)
            h2 = F.dropout2d(h1, 0.3)
            h2 = F.relu(F.max_pool2d(self.conv2(h1), kernel_size=2, stride=2), inplace=True)
            h2 = F.dropout2d(h2, 0.3)
            h3 = F.relu(self.fc1(h2.view((-1, 256))), inplace=True)
            h3 = F.dropout(h3, 0.3)
            h4 = F.relu(self.fc2(h3), inplace=True)
            y.append(h4)

        y1 = F.relu(self.lin4(torch.cat((y[0].view(-1, 84), y[1].view(-1, 84)), 1)), inplace=True)
        y2 = self.lin5(y1)
        return y2


def train(model1, model2, train_input, train_target, train_class, val_input, val_target, val_classes, mini_batch_size, nb_epochs=25, verbose=False):
    losses = torch.zeros(nb_epochs)
    val_losses = torch.zeros(nb_epochs)
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=1e-3)

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

        losses[e] = sum_loss
        val_losses[e] = get_loss_val(model2.eval(), val_input, val_target)
    return losses, val_losses


def get_loss_val(model, val_input, val_target):
    criterion = nn.CrossEntropyLoss()
    pred = model(val_input)
    return criterion(pred, val_target)

def accuracy(preds, targets):
    return (preds.argmax(axis=1) == targets).long().sum().item() / targets.shape[0]

def get_mis_class_aux(model, input_, target, classes):
    preds = model(input_)
    preds = preds.argmax(axis=1) == target
    misclassified = classes[~preds]
    return misclassified.tolist()

def train_all(train_input, train_target, train_classes, val_input, val_target, val_classes, test_input, test_target,
              test_classes, niter=15, nb_epochs=25, mini_batch_size=100):
    all_classified = []
    misclassified = []
    accuracies_train = []
    accuracies_test = []
    accuracies_val = []
    losses = torch.zeros((niter, nb_epochs))
    losses_val = torch.zeros((niter, nb_epochs))

    for i in range(niter):
        print("-" * 50, f" \n Iteration {i} \n ")

        # define the model
        model1 = BaseNet()
        model2 = Full_Net()

        # train model
        losses[i, :], losses_val[i, :] = train(model1.train(), model2.train(), train_input,train_target,train_classes,
                                               val_input, val_target,val_classes,mini_batch_size, nb_epochs=nb_epochs)
        model = model2.eval()
        train_accuracy = accuracy(model(train_input), train_target)
        test_accuracy = accuracy(model(test_input), test_target)
        val_accuracy = accuracy(model(val_input), val_target)

        misclass = get_mis_class_aux(model, torch.cat((test_input, val_input)), torch.cat((test_target, val_target)),
                                 torch.cat((test_classes, val_classes)))
        [all_classified.append(x) for x in torch.cat((test_classes, val_classes))]
        [misclassified.append(x) for x in misclass]
        accuracies_train.append(train_accuracy)
        accuracies_test.append(test_accuracy)
        accuracies_val.append(val_accuracy)

        print(f"Training accuracy is {train_accuracy} ")
        print(f"Validation accuracy is {val_accuracy} ")
        pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    return losses, losses_val, accuracies_train, accuracies_test, accuracies_val, all_classified, misclassified
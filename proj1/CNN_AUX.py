import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np


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
        # Image 1 class
        h1a = F.relu(F.max_pool2d(self.conv1(x[:,0,:,:].view(x.size(0),1,14,14)), kernel_size=2))
        h1a = F.dropout2d(h1a, 0.7)
        h2a = F.relu(F.max_pool2d(self.conv2(h1a), kernel_size=2, stride=2))
        #h2a = F.dropout2d(h2a, 0.7)
        h3a = F.relu(self.fc1(h2a.view((-1, 256))))
        #h3a = F.dropout(h3a, 0.7)
        h4a = F.relu(self.fc2(h3a))
        #h4a = F.dropout(h4a, 0.7)
        h5a = self.fc3(h4a)

        # Image 2 class
        h1b = F.relu(F.max_pool2d(self.conv1(x[:,1,:,:].view(x.size(0),1,14,14)), kernel_size=2))
        h1b = F.dropout2d(h1b, 0.7)
        h2b = F.relu(F.max_pool2d(self.conv2(h1b), kernel_size=2, stride=2))
        #h2b = F.dropout2d(h2b, 0.7)
        h3b = F.relu((self.fc1(h2b.view(-1,256))))
        #h3b = F.dropout(h3b, 0.7)
        h4b = F.relu(self.fc2(h3b))
        #h4b = F.dropout(h4b, 0.7)
        h5b = self.fc3(h4b)


        # Classifiction
        y1 = F.relu(self.lin4(torch.cat((h4a.view(-1, 84), h4b.view(-1, 84)), 1)))
        y2 = self.lin5(y1)
        return [y2, h5a, h5b]

def train(model, train_input, train_target, train_class, val_input, val_target, mini_batch_size, nb_epochs=25, verbose=False):
    losses = np.zeros(nb_epochs)
    val_losses = np.zeros(nb_epochs)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for e in range(nb_epochs):
        model = model.train()
        beta = (nb_epochs - e) / nb_epochs
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output[0], train_target.narrow(0, b, mini_batch_size))
            aux_loss = criterion(output[1], train_class[:, 0].narrow(0, b, mini_batch_size)) + \
                       criterion(output[2], train_class[:, 1].narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item() + aux_loss.item()
            model.zero_grad()
            ((1 - beta) * loss + beta * aux_loss).backward()
            optimizer.step()
        losses[e] = sum_loss
        if verbose:
            print(e, sum_loss)
        val_losses[e] = get_loss_val(model.eval(), val_input, val_target)
    return losses,val_losses


def get_loss_val(model, val_input, val_target):
    criterion = nn.CrossEntropyLoss()
    pred = model(val_input)[0]
    return criterion(pred, val_target)


def accuracy(preds, targets):
    return (preds.argmax(axis=1) == targets).long().sum().item() / targets.shape[0]


def get_mis_class_aux(model, input_, target, classes):
    preds = model(input_)[0]
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
    losses = np.zeros((niter, nb_epochs))
    losses_val = np.zeros((niter, nb_epochs))

    for i in range(niter):
        print("-" * 50, f" \n Iteration {i} \n ")

        # define the model
        model = CNN_aux()

        # train model
        losses[i, :], losses_val[i, :] = train(model.train(), train_input, train_target, train_classes,
                                               val_input, val_target, mini_batch_size, nb_epochs=nb_epochs)
        model = model.eval()
        train_accuracy = accuracy(model(train_input)[0], train_target)
        test_accuracy = accuracy(model(test_input)[0], test_target)
        val_accuracy = accuracy(model(val_input)[0], val_target)

        misclass = get_mis_class_aux(model, torch.cat((test_input, val_input)), torch.cat((test_target, val_target)),
                                 torch.cat((test_classes, val_classes)))
        [all_classified.append(x) for x in torch.cat((test_classes, val_classes))]
        [misclassified.append(x) for x in misclass]
        accuracies_train.append(train_accuracy)
        accuracies_test.append(test_accuracy)
        accuracies_val.append(val_accuracy)

        print(f"Training accuracy is {train_accuracy} ")
        print(f"Validation accuracy is {val_accuracy} ")
    return losses, losses_val, accuracies_train, accuracies_test, accuracies_val, all_classified, misclassified
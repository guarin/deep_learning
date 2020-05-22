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

def train(model, train_input, train_target, train_class, val_input, val_target, val_classes, mini_batch_size, nb_epochs=25, verbose=False):
    losses = torch.zeros(nb_epochs)
    val_losses = torch.zeros(nb_epochs)
    aux_losses = torch.zeros((nb_epochs))
    val_aux_losses = torch.zeros((nb_epochs))
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
            aux_loss = criterion(output[1], train_class[:, 0].narrow(0, b, mini_batch_size)) + \
                       criterion(output[2], train_class[:, 1].narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            sum_aux_loss = sum_aux_loss + aux_loss.item()
            model.zero_grad()
            ((1 - beta) * loss + beta * aux_loss).backward()
            optimizer.step()
        losses[e] = sum_loss
        aux_losses[e] = sum_aux_loss
        if verbose:
            print(e, sum_loss)
        val_losses[e] = get_loss_val(model.eval(), val_input, val_target)
        val_aux_losses[e] = get_aux_loss_val(model.eval(), val_input, val_classes)
    return losses, val_losses, aux_losses, val_aux_losses


def get_loss_val(model, val_input, val_target):
    criterion = nn.CrossEntropyLoss()
    pred = model(val_input)[0]
    return criterion(pred, val_target)

def get_aux_loss_val(model, val_input, val_classes):
    criterion = nn.CrossEntropyLoss()
    pred_1 = model(val_input)[1]
    pred_2 = model(val_input)[2]
    return criterion(pred_1, val_classes[:, 0]) + criterion(pred_2, val_classes[:, 1])

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
    losses = torch.zeros((niter, nb_epochs))
    losses_val = torch.zeros((niter, nb_epochs))
    aux_losses = torch.zeros((niter, nb_epochs))
    val_aux_losses = torch.zeros((niter, nb_epochs))

    for i in range(niter):
        print("-" * 50, f" \n Iteration {i} \n ")

        # define the model
        model = CNN_aux()

        # train model
        losses[i, :], losses_val[i, :], aux_losses[i, :], val_aux_losses[i,:] = train(model.train(), train_input,
                                                                                      train_target,train_classes,
                                                                                      val_input, val_target,val_classes,
                                                                                      mini_batch_size, nb_epochs=nb_epochs)
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
        pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    return losses, losses_val,aux_losses, val_aux_losses, accuracies_train, accuracies_test, accuracies_val, all_classified, misclassified
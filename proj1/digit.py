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
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view((-1, 256))))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
######################################################################################

def aux_loss(output_1,output_2,classes,criterion):
    # loss of prediction of the two number
    loss_1 = criterion(output_1, classes[:,0])
    loss_2 = criterion(output_2,  classes[:,1])
    
    return loss_1, loss_2


# Training function
# -----------------------------------------------------------------------------------
def train(model, train_input, train_target, train_classes, mini_batch_size,nb_epochs=25, verbose = False):
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    for e in range(nb_epochs):
        model = model.train()
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            optimizer.zero_grad()
            output_1 = model(train_input.narrow(0, b, mini_batch_size)[:,0].unsqueeze(1))
            output_2 = model(train_input.narrow(0, b, mini_batch_size)[:,1].unsqueeze(1))
            loss_1,loss_2  = aux_loss(output_1,output_2,train_classes.narrow(0, b, mini_batch_size),criterion)
            loss = loss_1 + loss_1
            sum_loss = sum_loss + loss.item()
            loss.backward()
            optimizer.step()
        if verbose:
            print(e,sum_loss)


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
    # Predict class of inputs
    model.eval()
    output_1 = model(inputs[:,0].unsqueeze(1))
    output_2 = model(inputs[:,1].unsqueeze(1))
    preds = output_1.argmax(dim=1) <= output_2.argmax(dim=1)
    # Compute accuracy
    accuracy = (preds == targets).long().sum().item()/len(preds)
    return accuracy 


######################################################################################

def get_model():
    return BaseNet()
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim


class LeNetLike(nn.Module):
    """ Modified LeNet. Takes input format 1 x 14 x 14 and outputs the class"""
    def __init__(self):
        super(LeNetLike, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size = 3),
            nn.ReLU()
)
        self.classifier = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1)
)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
######################################################################################


# Accuracy for the baseline trains to get the classes then compare predicted class
def accuracy(model,inputs,targets):
    """ INPUT : 
        - model: model that predicts the digit values 
        - input_long: the input of the format 2000 x 10
        - tagets: ground truth of the pairs comparaison with format 1000 """
    
    # get long inputs
    inputs_long = torch.cat((inputs[:,0,...],inputs[:,1,...]),0).unsqueeze(1)
    # Predict class of inputs
    preds_long = model(inputs_long)
    # Compare the pairs
    preds = (preds_long[:1000,:] <= preds_long[1000:,:]).long().squeeze()
    # Compute accuracy
    accuracy = (preds == targets).long().sum().item()/len(preds)
    return accuracy 

def train(model, train_input, train_target, mini_batch_size, verbose = False):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    for e in range(25):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            optimizer.zero_grad()
            inputs = train_input.narrow(0, b, mini_batch_size)
            output_1 = model(inputs[:,0,...].unsqueeze(1))
            output_2  = model(inputs[:,1,...].unsqueeze(1))
            preds = F.sigmoid(output_2 - output_1).squeeze()
            loss = criterion(preds, train_target.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            loss.backward()
            optimizer.step()
        if verbose:
            print(e,sum_loss)
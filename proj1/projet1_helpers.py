import torch
from torch import nn
from torch import optim
import numpy as np
import proj1.dlc_practical_prologue as prologue
import matplotlib.pyplot as plt

def plot_performance (accuracies, n_iter):
    x = np.linspace(0,n_iter,n_iter)
    plt.plot(x,accuracies,'k-')
    error = np.sqrt(np.var(accuracies))
    plt.fill_between(x, accuracies-error, accuracies+error)
           
        
######################################################################################

def accuracy_one_hot(preds, target):
    return (preds.argmax(axis=1) == target.argmax(axis=1)).long().sum().item() / target.shape[0]

def load_data(n):
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(n)
    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)
    return train_input, train_target, train_classes, test_input, test_target, test_classes
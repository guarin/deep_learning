import torch
from torch import nn
from torch import optim
import numpy as np

import matplotlib.pyplot as plt

def plot_performance (accuracies, n_iter):
    x = np.linspace(0,n_iter,n_iter)
    plt.plot(x,accuracies,'k-')
    error = np.sqrt(np.var(accuracies))
    plt.fill_between(x, accuracies-error, accuracies+error)
           
        
######################################################################################

def accuracy_one_hot(preds, target):
    return (preds.argmax(axis=1) == target.argmax(axis=1)).long().sum().item() / target.shape[0]
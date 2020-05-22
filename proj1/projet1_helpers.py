import torch
from torch import nn
from torch import optim
import numpy as np
import pandas as pd
import dlc_practical_prologue as prologue

import matplotlib.pyplot as plt

def plot_performance(accuracies_train,accuracies_test):
    plt.figure(figsize=(8,8))
    plt.boxplot([accuracies_train,accuracies_test],labels = ["train","test"])
    plt.ylabel('accuracy', fontsize=16)
    plt.show()
           
        
######################################################################################

def accuracy_one_hot(preds, target):
    return (preds.argmax(axis=1) == target.argmax(axis=1)).long().sum().item() / target.shape[0]

def plot_heatmap(classes,normalize):
    df_norm = pd.DataFrame(normalize, columns=['x', 'y'])
    df_norm['count'] = 1
    heat_norm = df_norm.groupby(['x', 'y']).count().reset_index().pivot(index='x', columns='y', values='count').fillna(0)
    df = pd.DataFrame(classes, columns=['x', 'y'])
    df['count'] = 1
    heat = df.groupby(['x', 'y']).count().reset_index().pivot(index='x', columns='y', values='count').fillna(0)
    heat = heat/heat_norm
    fig = plt.figure()
    plt.pcolor(heat)
    plt.yticks(np.arange(0.5, len(heat.index), 1), heat.index)
    plt.xticks(np.arange(0.5, len(heat.columns), 1), heat.columns)
    plt.ylabel('First Value')
    plt.xlabel('Second Value')
    plt.title("Misclassification per digit pairs")
    plt.colorbar(label = "Misclassification rate")
    fig.savefig("Plots/Heatmap.png")
    plt.show()
    return heat,fig

def get_mis_class(model,input_,target,classes):
    preds = model(input_).round() == target
    misclassified = classes[~preds]
    return misclassified.tolist()

def get_mis_class_aux(model, input_, target, classes):
    preds = model(input_)[0].round() == target
    misclassified = classes[~preds]
    return  misclassified.tolist()

def plotloss(losses,color):
    mean = losses.mean(axis=1)
    error = np.sqrt(losses.var(axis=1))
    x = np.linspace(0, len(mean), len(mean))
    plt.title('Loss Change across Rounds')
    plt.ylabel('Loss')
    plt.xlabel('# Round')
    plt.plot(x, mean, color+"-")
    plt.fill_between(x, mean-error, mean+error,color=color,alpha = 0.5)
    
def load_data(n):
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(n)
    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)
    return train_input, train_target, train_classes, test_input, test_target, test_classes

def load_test_data():
    train_input, train_target, train_classes, test_input, test_target, test_classes = load_data(1000)
    val_input = test_input[:500]
    test_input = test_input[500:]
    val_target = test_target[:500]
    test_target = test_target[500:]
    val_classes = test_classes[:500]
    test_classes = test_classes[500:]
    return train_input, train_target, train_classes,val_input,test_input,val_target,test_target,val_classes,test_classes
import torch
from torch import nn
from torch import optim
import numpy as np
import pandas as pd

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
    plt.pcolor(heat)
    plt.yticks(np.arange(0.5, len(heat.index), 1), heat.index)
    plt.xticks(np.arange(0.5, len(heat.columns), 1), heat.columns)
    plt.colorbar()
    plt.show()
    
def get_mis_class(model,input_,target,classes):
    preds = model(input_).round() == target
    misclassified = classes[~preds]
    return misclassified.tolist()
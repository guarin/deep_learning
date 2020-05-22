import baseline as bl
import dlc_practical_prologue as prologue
import CNN_reg as cnn
import CNN_WS as ws
import CNN_AUX as aux
import digit as dig
import transfer_learning as tl
import torch
import sys
import argparse


# Load Data function
# -----------------------------------------------------------------------------------
def load_data(n):
    """ INPUT:
        - n: Number of training and test points
        OUTPUT:
        - train_input:      Training images
        - train_target:     Training binary targets
        - train_classes:    Training digit classifications
        - test_input:       Test images
        - test_target:      Test binary targets
        - test_classes:     Test digit classifications
    """
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(n)
    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)
    return train_input, train_target, train_classes, test_input, test_target, test_classes


# Training function
# -----------------------------------------------------------------------------------
def train_all(architecture,train_input, train_target, train_classes, test_input, test_target, test_classes, niter = 15, nb_epochs = 25, mini_batch_size = 100):
    """ INPUT:
        - architecture:     Network architecture to be evaluated
        - train_input:      Training images
        - train_target:     Training binary targets
        - train_classes:    Training digit classifications
        - test_input:       Test images
        - test_target:      Test binary targets
        - test_classes:     Test digit classifications
        - niter:            Iterations to evaluate the model
        - nb_epochs:        Training iterations
        - mini_batch_size:  Size of the Mini Batch
        OUTPUT:
        None
    Takes a given Network, trains it on the training data and evaluates on the test data. Prints summary statistics
    of the evaluations
    """
    print("-" * 50, f" \n Iterations {niter}, {architecture[1]}")
    accuracies_train, accuracies_test = torch.zeros(niter), torch.zeros(niter)

    for i in range(niter):
        model = architecture[0].get_model()
        # train model
        architecture[0].train(model, train_input, train_target, train_classes, mini_batch_size, nb_epochs=nb_epochs, verbose = False)

        accuracies_train[i] = architecture[0].accuracy(model,train_input,train_target)
        accuracies_test[i] = architecture[0].accuracy(model,test_input,test_target)

    print(f"The mean train accuracy of the {architecture[1]} is {accuracies_train.mean():.4f} ± {accuracies_train.var():.4f} ")
    print(f"The mean test accuracy of the {architecture[1]} is {accuracies_test.mean():.4f} ± {accuracies_test.var():.4f} ")


# Run Models function
# -----------------------------------------------------------------------------------
def run_models():
    """
    Defines all architectures and evaluates them on generated training and test data
    """
    torch.manual_seed(42)
    train_input, train_target, train_classes, test_input, test_target, test_classes = (load_data(1000))

    architectures = [(bl, "Baseline Model"),
                     (cnn, "CNN Model"),
                     (ws, "CNN Weight Sharing Model"),
                     (aux, "CNN Auxiliary Loss Model"),
                     (tl, "Transfer learning Model"),
                     (dig, "Digit Classification Model")]

    for architecture in architectures:
        train_all(architecture, train_input, train_target, train_classes, test_input, test_target, test_classes, nb_epochs=25)


if __name__ == "__main__":
    print('Training 6 different architectures 15 times with 25 epochs, this may take some time')
    run_models()
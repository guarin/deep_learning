from framework import nn
from framework.util import batch_train, batch_test, compute_nb_errors
import util
import torch


def test_network():
    return nn.Sequential(
        nn.Linear(2, 25, initialization='He'),
        nn.ReLU(),
        nn.Linear(25, 25, initialization='He'),
        nn.ReLU(),
        nn.Linear(25, 2, initialization='He')
    )


def test_network_dropout():
    return nn.Sequential(
        nn.Linear(2, 25, initialization='He'),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(25, 25, initialization='He'),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(25, 2, initialization='He')
    )


#  Dictionary of all tests
#  values are: (network, criterion, optimizer, learning_rate)
tests = {
    'MSE-SGD': (test_network, nn.MSELoss, nn.SGD, 0.1),
    'MSE-Adam': (test_network, nn.MSELoss, nn.Adam, 0.01),
    'MSE-Adam-Dropout': (test_network_dropout, nn.MSELoss, nn.Adam, 0.01),
    'CrossEntropy-SGD': (test_network, nn.CrossEntropyLoss, nn.SGD, 0.1),
    'CrossEntropy-Adam': (test_network, nn.CrossEntropyLoss, nn.Adam, 0.01),
    'CrossEntropy-Adam-Dropout': (test_network_dropout, nn.CrossEntropyLoss, nn.Adam, 0.01),
}


def run_test(n_samples=1000, grad=False, show_plot=True, save_plot=False, data_seed=0, model_seed=1):
    torch.set_grad_enabled(grad)

    torch.manual_seed(data_seed)
    train_input, train_target = util.generate_samples(n_samples)
    test_input, test_target = util.generate_samples(n_samples)

    train_results = dict()
    test_results = dict()
    for name, (model, criterion, optimizer, learning_rate) in tests.items():
        torch.manual_seed(model_seed)
        model = model()
        criterion = criterion()
        optimizer = optimizer(model.parameters, learning_rate=learning_rate)
        print(name)
        print('Train ' + ('-' * (42 - len('Train '))))
        train_results[name] = batch_train(model, criterion, optimizer, train_input, train_target, verbose=True,
                                          nb_errors=True)
        model.eval()
        print('Test ' + ('-' * (42 - len('Test '))))
        test_results[name] = batch_test(model, criterion, test_input, test_target, verbose=True, nb_errors=True)
        print('=' * 42)
        print('')

    if show_plot:
        import matplotlib.pyplot as plt
        train_errors = {
            name: [100 * compute_nb_errors(predictions.argmax(1), train_target) / len(train_target) for _, predictions
                   in values] for name, values in train_results.items()}
        [plt.plot(range(len(values)), values, label=name) for name, values in train_errors.items()];
        plt.legend()
        plt.xlim((0, 100))
        plt.xlabel('Epoch')
        plt.ylabel('Error Percentage')
        if save_plot:
            plt.savefig('train_error.png')
        plt.show()


if __name__ == '__main__':
    run_test(show_plot=False)

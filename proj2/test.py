from framework import nn
from framework.util import batch_train, batch_test
import util
import torch


def run_test(n_samples=1000, grad=False):
	torch.set_grad_enabled(grad)

	train_input, train_target = util.generate_samples(n_samples)
	print(train_target.float().mean())
	test_input, test_target = util.generate_samples(n_samples)
	model = nn.Sequential(
		nn.Linear(2, 25),
		nn.ReLU(),
		nn.Linear(25, 25),
		nn.ReLU(),
		nn.Linear(25, 2)
	)
	criterion = nn.MSELoss()
	optimizer = nn.SGD(model.parameters, learning_rate=0.1, momentum=0.1)
	print('Train')
	batch_train(model, criterion, optimizer, train_input, train_target, verbose=True, nb_errors=True)
	print('-'*20)
	print('Test')
	batch_test(model, criterion, test_input, test_target, verbose=True, nb_errors=True)


if __name__ == '__main__':
	run_test()

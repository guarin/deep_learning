from framework import nn
import util
import torch


def run_test(n_samples=1000, grad=False):
	torch.set_grad_enabled(grad)

	train_input, train_label = util.generate_samples(n_samples)
	test_input, test_label = util.generate_samples(n_samples)
	torch.autograd.gradcheck()


if __name__ == '__main__':
	run_test()

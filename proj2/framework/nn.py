"""Modules for custom deep learning framework

Modules mirror the syntax from pytorch
"""

from torch import empty
from torch import empty_like
from framework import function
from framework import util
import math


class Parameter:
    """Parameter of a Module

    Attributes
    ----------
    name : str
            Name of the parameter
    value : Tensor (n, m)
            Value of the parameter
    gradient : Tensor (n, m)
            Gradient of the parameter
    """

    def __init__(self, value, name=None):
        self.name = name
        self.value = value
        self.gradient = empty(value.shape).zero_()

    def __repr__(self):
        return f"Parameter({self.value.shape}, {self.name!r})"


class Module:
    """Base class for all modules"""

    def __init__(self):
        self.training = True

    def forward(self, input):
        raise NotImplementedError

    def backward(self, gradient):
        raise NotImplementedError

    def parameters(self, recurse=True):
        """Iterator over model parameters"""
        return iter(())

    def zero_grad(self):
        """Runs zero_grad on all parameters in the module"""
        for parameter in self.parameters():
            parameter.gradient.zero_()

    def train(self, mode=True):
        """Sets the module into training mode = `mode`"""
        self.training = mode
        return self

    def eval(self):
        """Sets the model into evaluation mode"""
        return self.train(False)

    def __call__(self, *args):
        """Allows module.forward(input) to be called like module(input)"""
        return self.forward(*args)


class Linear(Module):
    """Fully connected layer using Xavier normal initialization"""

    initialization_gain = {
        'Xavier': 1.0,
        'He': math.sqrt(2.0)
    }

    def __init__(self, in_size, out_size, initialization='Xavier'):
        super().__init__()
        weight = function.xavier_normal_(empty(out_size, in_size), gain=self.initialization_gain[initialization])
        bias = empty(out_size).zero_()
        self.weight = Parameter(weight, 'weight')
        self.bias = Parameter(bias, 'bias')
        self._input = None

    def forward(self, input):
        self._input = input
        return input @ self.weight.value.T + self.bias.value

    def backward(self, gradient):
        self.weight.gradient += gradient.T @ self._input
        self.bias.gradient += gradient.sum(0)
        return gradient @ self.weight.value

    def parameters(self, recurse=True):
        yield self.weight
        yield self.bias


class Sequential(Module):
    """Sequential module

    Attributes
    ----------
    modules : List(Module)
    """

    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, input):
        output = input
        for module in self.modules:
            output = module.forward(output)
        return output

    def backward(self, gradient):
        for module in reversed(self.modules):
            gradient = module.backward(gradient)
        return gradient

    def parameters(self, recurse=True):
        if recurse:
            for module in self.modules:
                for parameter in module.parameters():
                    yield parameter
        else:
            return iter(())

    def train(self, mode):
        for module in self.modules:
            module.train(mode)
        return self


class Dropout(Module):
    """Dropout module"

    Attributes
    ----------
    p : float
        Probability for activation to be ignored
    """

    def __init__(self, p=0.5):
        super().__init__()
        from torch.distributions import Bernoulli
        self.p = p
        self._bernoulli = Bernoulli(1 - self.p)
        self._sample = None

    def forward(self, input):
        if self.training:
            self._sample = self._bernoulli.sample(input.shape).float() / (1 - self.p)
            input = input * self._sample
        return input

    def backward(self, gradient):
        if self.training:
            gradient = gradient * self._sample
        return gradient


class ReLU(Module):
    """Rectified Linear Unit"""

    def forward(self, input):
        self._input = input
        return input * (input > 0).float()

    def backward(self, gradient):
        return (self._input > 0).float() * gradient


class Tanh(Module):
    """Hyperbolic tangent module"""

    def forward(self, input):
        self._input = input
        return function.tanh(input)

    def backward(self, gradient):
        return (1 - function.tanh(self._input).pow(2)) * gradient


class LogSoftmax(Module):
    """Logarithmic softmax module"""

    def forward(self, input):
        self._softmax = function.softmax(input)
        return self._softmax.log()

    def backward(self, gradient):
        return gradient - gradient.sum(1).view(-1, 1) * self._softmax


class Loss(Module):
    """Base class for all losses"""

    def forward(self, input, target):
        raise NotImplementedError


class MSELoss(Loss):
    """Mean Squared Error Loss

    Detects automatically one hot encoded inputs
    """

    def forward(self, input, target):
        if input.shape != target.shape:
            target = util.one_hot_encode(n_classes=input.shape[1], target=target)
        self._diff = input - target
        return function.mse(input, target)

    def backward(self):
        return 2 * self._diff / self._diff.nelement()


class NLLLoss(Loss):

    def forward(self, input, target):
        """Target is assumed to contain class indices"""
        self._target = target
        self._shape = input.shape
        # gather selects only values in input at the indices given by target
        return -input.gather(1, target.view(-1, 1)).mean()

    def backward(self):
        from torch import arange
        gradient = empty(self._shape).zero_()
        gradient[arange(self._shape[0]), self._target] = -1.0 / self._shape[0]
        return gradient


class CrossEntropyLoss(Loss):
    """Cross Entropy Loss implemented as sequence of LogSoftmax layer followed by NLLLoss"""

    def __init__(self):
        super().__init__()
        self._log_softmax = LogSoftmax()
        self._nllloss = NLLLoss()

    def forward(self, input, target):
        output = self._log_softmax.forward(input)
        loss = self._nllloss.forward(output, target)
        return loss

    def backward(self):
        gradient = self._nllloss.backward()
        return self._log_softmax.backward(gradient)

    def zero_grad(self):
        self._log_softmax.zero_grad()
        self._nllloss.zero_grad()


class Optimizer:
    """Base class for all optimizers"""

    def __init__(self, parameters):
        self.parameters = parameters

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent supporting momentum"""

    def __init__(self, parameters, learning_rate=0.1, momentum=0):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.momentum = momentum
        if momentum > 0:
            # initialize momentum for each parameter
            self._momentum = [empty_like(p.value).zero_() for p in parameters()]

    def step(self):
        for i, parameter in enumerate(self.parameters()):
            u = parameter.gradient
            if self.momentum > 0:
                u += self.momentum * self._momentum[i]
                self._momentum[i].copy_(u)
            u *= self.learning_rate
            parameter.value -= u


class Adam(Optimizer):
    """Classic Adam implementation"""

    def __init__(self, parameters, learning_rate=0.001, betas=(0.9, 0.999), eps=1e-08):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self._m = [empty_like(p.value).zero_() for p in parameters()]
        self._v = [empty_like(p.value).zero_() for p in parameters()]

    def step(self):
        for parameter, m, v in zip(self.parameters(), self._m, self._v):
            m.copy_(self.betas[0] * m + (1 - self.betas[0]) * parameter.gradient)
            v.copy_(self.betas[1] * v + (1 - self.betas[1]) * parameter.gradient.pow(2))
            gradient = self.learning_rate / ((v / (1.0 - self.betas[1])).sqrt() + self.eps) * (m / (1.0 - self.betas[0]))
            parameter.value -= gradient

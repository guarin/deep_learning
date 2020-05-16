from unittest import TestCase
import torch
from framework import nn


def generate_xxy():
    """Generates random tensors used for regression testing

    Returns
    -------
    x : Tensor (3, 5)
    x_req_grad : Tensor (3, 5), requires gradient
    y : Tensor (3, 5)
    """
    x = torch.empty((3, 5)).normal_(0, 1)
    x_req_grad = x.clone().requires_grad_()
    y = torch.empty((3, 5)).normal_(0, 1)
    return x, x_req_grad, y


def generate_class_xxy():
    """Generates random tensors used for classification testing

    Returns
    -------
    x : Tensor (3, 5)
    x_req_grad : Tensor (3, 5), requires gradient
    y : Tensor (3, 1)
    """
    x = torch.empty((3, 5)).normal_(0, 1)
    x_req_grad = x.clone().requires_grad_()
    y = x.argmin(1)
    return x, x_req_grad, y


def parameters_normal_(model, seed=1):
    """Instantiates all model parameters to have normal distribution with mean 0 and variance 1"""
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            if isinstance(parameter, nn.Parameter):
                parameter.value.normal_(0, 1)
            else:
                parameter.normal_(0, 1)


class ForwardBackwardTest(TestCase):
    """Base test class that automatically runs a forward and a backward pass for
    `module` and verifies that the results are identical to a forward and backward
    pass for `torch_module``

    Attributes
    ----------
    torch_module : torch.nn.Module
    module : framework.nn.Module
    """
    torch_module = None
    module = None

    def setUp(self, loss=False, classification=False):
        """Initializes the class for testing"""
        if classification:
            self.x, self.x_req_grad, self.y = generate_class_xxy()
        else:
            self.x, self.x_req_grad, self.y = generate_xxy()
        self.loss = loss
        self.forward()
        self.backward()

    def forward(self):
        """Runs the forward pass"""
        if self.loss:
            self.torch_output = self.torch_module.forward(self.x_req_grad, self.y)
            self.output = self.module.forward(self.x, self.y)
        else:
            self.torch_output = self.torch_module.forward(self.x_req_grad)
            self.output = self.module.forward(self.x)

    def backward(self):
        """Runs the backward pass"""
        self.torch_output.sum().backward()
        self.torch_x_grad = self.x_req_grad.grad
        if self.loss:
            self.x_grad = self.module.backward()
        else:
            self.x_grad = self.module.backward(torch.ones_like(self.output))

    def test_forward(self):
        """Tests whether outputs of forward pass are identical"""
        self.assertTrue(torch.allclose(self.output, self.torch_output),
                        f"Outputs not equal {self.output}, {self.torch_output}")

    def test_backward(self):
        """Tests whether final gradients of modules are equal and whether
        all parameters are equal after backward pass."""
        self.assertTrue(torch.allclose(self.x_grad, self.torch_x_grad),
                        f"Input gradients not equal {self.x_grad}, {self.torch_x_grad}")
        for (torch_parameter, parameter) in zip(self.torch_module.parameters(), self.module.parameters()):
            self.assertTrue(torch.allclose(torch_parameter, parameter.value))


class TestLinear(ForwardBackwardTest):
    def setUp(self):
        self.in_size = 5
        self.out_size = 4
        self.torch_module = torch.nn.Linear(self.in_size, self.out_size)
        self.module = nn.Linear(self.in_size, self.out_size)
        # explicitly set parameters to identical values
        parameters_normal_(self.torch_module)
        parameters_normal_(self.module)
        super().setUp()


class TestSequential(ForwardBackwardTest):
    def setUp(self):
        self.in_size = 5
        self.mid_size = 6
        self.out_size = 5
        self.torch_module = torch.nn.Sequential(
            torch.nn.Linear(self.in_size, self.mid_size),
            torch.nn.Linear(self.mid_size, self.out_size)
        )
        self.module = nn.Sequential(
            nn.Linear(self.in_size, self.mid_size),
            nn.Linear(self.mid_size, self.out_size)
        )
        parameters_normal_(self.torch_module)
        parameters_normal_(self.module)
        super().setUp()


class TestReLU(ForwardBackwardTest):
    def setUp(self):
        self.module = nn.ReLU()
        self.torch_module = torch.nn.ReLU()
        super().setUp()


class TestTanh(ForwardBackwardTest):
    def setUp(self):
        self.torch_module = torch.nn.Tanh()
        self.module = nn.Tanh()
        super().setUp()


class TestLogSoftmax(ForwardBackwardTest):
    def setUp(self):
        self.torch_module = torch.nn.LogSoftmax()
        self.module = nn.LogSoftmax()
        super().setUp()


class TestMSELoss(ForwardBackwardTest):
    def setUp(self):
        self.torch_module = torch.nn.MSELoss()
        self.module = nn.MSELoss()
        super().setUp(loss=True)


class TestNLLLoss(ForwardBackwardTest):
    def setUp(self):
        self.torch_module = torch.nn.NLLLoss()
        self.module = nn.NLLLoss()
        super().setUp(loss=True, classification=True)


class TestCrossEntropyLoss(ForwardBackwardTest):
    def setUp(self):
        self.torch_module = torch.nn.CrossEntropyLoss()
        self.module = nn.CrossEntropyLoss()
        super().setUp(loss=True, classification=True)

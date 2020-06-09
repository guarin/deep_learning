# Deep Learning Projects

## Project 1 - Digit Classification
Binary classification on Mnist pairs of handwritten digits

[Code](proj1)

The module contains six different architectures that gradually contain more information about the task.
By running the test.py script without any arguments, the different models get evaluated 15 times on 25 epochs.

Better performance is reached if the epochs are increased to 100.


## Project 2 - Deep Learning Framework
Implementation of a deep learning framework based on PyTorch without using the autograd functionality. Provides basic modules such as: Linear, Sequence, DropOut, LogSoftmax, MSELoss and CrossEntropyLoss. SGD and Adam optimzer are implemented as well.

[Code](proj2)

The module contains the framework and a test file which runs six basic architectures. An additional test file is provided in `/framework/test_nn.py` that runs tests comparing the functionality of the implemented modules with the original PyTorch modules.

def to_class(prediction):
    """Returns class from a prediction"""
    return prediction.argmax(1)


def one_hot_encode(n_classes, target):
    """One hot encodes the tensor `target` to `n_classes`

    Parameters
    ----------
    n_classes : int
                Number of classes
    target : Tensor (n,)
                Tensor to encode

    Returns
    -------
    one_hot : Tensor (n, n_classes)
    """
    from torch import empty, arange
    n = target.shape[0]
    one_hot = empty((n, n_classes)).zero_()
    one_hot[arange(n), target] = 1
    return one_hot


def split_batches(train, target, batch_size):
    """Iterator that splits `train` and `target` into batches of size `batch_size`

    Parameters
    ----------
    train : Tensor (n, m)
    target : Tensor (n, l)
    batch_size : int

    Returns
    -------
    train_batch : Tensor (batch_size, m)
    test_batch : Tensor (batch_size, l)
    """
    for i in range(0, train.size(0), batch_size):
        train_batch = train.narrow(0, i, batch_size)
        target_batch = target.narrow(0, i, batch_size)
        yield train_batch, target_batch


def forward_pass(model, criterion, input, target):
    """Runs single forward pass, returns loss and prediction

    Returns
    -------
    loss : float
    prediction : Tensor
    """
    prediction = model.forward(input)
    loss = criterion(prediction, target).item()
    return loss, prediction


def backward_pass(model, criterion, optimizer):
    """Runs single backward pass"""
    gradient = criterion.backward()
    model.zero_grad()
    model.backward(gradient)
    optimizer.step()


def train(model, criterion, optimizer, train_input, train_target, verbose=False, nb_errors=False):
    """Runs forward and backward pass

    Returns
    -------
    loss : float
    prediction : Tensor
    """
    loss, prediction = test(model, criterion, train_input, train_target, verbose, nb_errors)
    backward_pass(model, criterion, optimizer)
    return loss, prediction


def batch_train(model, criterion, optimizer, train_input, train_target, epochs=250, batch_size=100, verbose=False, nb_errors=False):
    """Trains model in batch mode"""
    results = []
    for epoch in range(epochs):
        epoch_results = []
        for train_batch, target_batch in split_batches(train_input, train_target, batch_size):
            train_result = train(model, criterion, optimizer, train_batch, target_batch)
            epoch_results.append(train_result)
        loss, prediction = concatenate_results(epoch_results)
        results.append((loss, prediction))
        print_result(loss, prediction, train_target, verbose, nb_errors, epoch)
    return results


def test(model, criterion, test_input, test_target, verbose=False, nb_errors=False):
    """Evaluates model for the given `test_input`

    Returns
    -------
    loss : float
    prediction : Tensor
    """
    loss, prediction = forward_pass(model, criterion, test_input, test_target)
    print_result(loss, prediction, test_target, verbose, nb_errors)
    return loss, prediction


def batch_test(model, criterion, test_input, test_target, batch_size=100, verbose=False, nb_errors=False):
    """Evaluates the model for the give `test_input` in batch mode

    Returns
    -------
    loss : float
    prediction : Tensor
    """
    test_results = []
    for test_batch, target_batch in split_batches(test_input, test_target, batch_size):
        test_result = test(model, criterion, test_batch, target_batch)
        test_results.append(test_result)
    loss, prediction = concatenate_results(test_results)
    print_result(loss, prediction, test_target, verbose, nb_errors)
    return loss, prediction


def compute_nb_errors(predicted_class, target_class):
    """Computes the number of errors between the `predicted_class` and the `target_class`"""
    return (predicted_class != target_class).long().sum().item()


def concatenate_results(results):
    """Concatenates a list of `results`

    Parameters
    ----------
    results : List((float, Tensor))

    Returns
    -------
    loss : float
    prediction : Tensor
    """
    from torch import cat
    loss = sum([result[0] for result in results])
    prediction = cat([result[1] for result in results])
    return loss, prediction


def print_result(loss, prediction, target, verbose=False, nb_errors=False, epoch=None):
    """Prints intermediate results during testing/training phase"""
    if verbose:
        messages = []
        if epoch is not None:
            messages.append(f"Epoch: {epoch:>3}")
        messages.append(f"Loss: {loss:7.4f}")
        if nb_errors:
            predicted_class = to_class(prediction)
            errors = compute_nb_errors(predicted_class, target)
            n = target.shape[0]
            mean_error = errors / n
            messages.append(f" Error: {mean_error*100:>7.3f}%")
        print(" ".join(messages))

import baseline as bl
import dlc_practical_prologue as prologue
import CNN_reg as cnn
import CNN_WS as ws
import CNN_AUX as aux
import digit as dig
import transfer_learning as tl
import torch

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

def run_models():
    torch.manual_seed(42)
    train_input, train_target, train_classes,val_input,test_input,val_target,test_target,val_classes,test_classes = (
        load_test_data()
    )

    architectures = [(cnn, "CNN"), (ws, "CNN with weight sharing"),
                     (tl, "Transfer learning"), (dig, "Digit prediction")]

    print("Baseline " + "accuracies")
    bl.train_all(train_input, train_target, train_classes, val_input, val_target, val_classes, test_input,
                               test_target, test_classes, nb_epochs=25)

    for (architecture, arch_name) in architectures:
        print(arch_name+" accuracies")
        architecture.train_all(train_input, train_target, train_classes,val_input, val_target, val_classes, test_input,
                                   test_target, test_classes,nb_epochs=25)

    print("Auxiliary Loss " + "accuracies")
    aux.train_all(train_input, train_target, train_classes, val_input, val_target, val_classes, test_input,
                               test_target, test_classes, nb_epochs=25)

if __name__ == "__main__":
    print('Training 6 different architectures, this may take some time')
    run_models()
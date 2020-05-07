import torch
import sys
sys.path.insert(0, '..')
import myutils

from torch import nn
from efficientnet_pytorch import EfficientNet


def efficientnetb7(unfreeze=False):
    """
    Unfreeze(True) all the model weights.
    Freeze(False) the convolutional layers only.
    """
    model = EfficientNet.from_pretrained('efficientnet-b7')
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, 8)

    class Net(nn.Module):


        def __init__(self):
            super().__init__()
            self.model = model


        def forward(self, x):
            x = self.model(x)
            x = nn.functional.softmax(x)
            return x

    net = Net()

    for param in model.parameters():
        param.requires_grad = unfreeze
    for param in model._fc.parameters():
        param.requires_grad = True
    return net


def loss_fn(weight):
    criterion = nn.CrossEntropyLoss(weight=weight)
    return criterion


if __name__ == '__main__':
    net = efficientnetb7()
    print(net)
    total_params, total_trainable_params = myutils.get_num_parameters(net)
    print('Total: {:,}\tTrainable: {:,}'.format(total_params, total_trainable_params))

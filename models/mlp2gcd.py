import torch.nn as nn
import torch
import torch.nn.functional as F
class two_layer_mlp(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(two_layer_mlp, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_dim, 124)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(124, 124)
        self.fc3 = nn.Linear(124, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class two_layer_mlp_dropout(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(two_layer_mlp_dropout, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(nn.Linear(input_dim, 124),
                                    nn.Dropout(p=0.3),
                                    nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Linear(124, 124),
                                    nn.Dropout(p=0.3),
                                    nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Linear(124, self.num_classes),
                                    )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
class LR(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(LR, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(nn.Linear(input_dim, self.num_classes))
        #self.layer2 = nn.Sequential(nn.Linear(64, 2))
    def forward(self, x):
        x = self.layer1(x)
        return x

def _test_mlp():
    net = two_layer_mlp(59)
    x = torch.zeros((1, 59))
    y = net(x)
    print(y.size())


if __name__ == '__main__':
    _test_mlp()
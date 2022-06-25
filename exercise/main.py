import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return x

    def aaa(self, x):
        print(self)
        x = self(x)
        return x


net = CNN()
a = torch.randn(1,3,3,3)
print(a)
res = net.aaa(a)
print(res)


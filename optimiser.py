import numpy as np
from network import NeuroNetwork

class SGD(object):
    def __init__(self, network:NeuroNetwork, lr):
        self.network = network
        self.lr = lr
    
    def forward(self, x):
        X = x
        for i in range(len(self.network.layers)- 1):
                X = self.network.layers[i].forward(X)
        return X

    def forward_with_loss(self, x, y):
        X = self.forward(x)
        return X, self.network.layers[-1].forward(X, y)
    
    def backward(self, y_, y):
        for i in range(len(self.network.layers) - 1, -1, -1):
            if i == len(self.network.layers) - 1:
                X = self.network.layers[i].backward(y_, y)
            else:
                X = self.network.layers[i].backward(X)
    
    def step(self):
        for i in range(len(self.network.layers)):
            self.network.layers[i].step(self.lr)

    def train(self, x, y):
        y_, ret = self.forward_with_loss(x, y)
        self.backward(y_, y)
        self.step()
        return ret
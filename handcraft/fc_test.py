import numpy as np
from network import *
from functions import *
from optimiser import *
from layers import *

x = np.random.rand(1000, 3)
y = np.eye(2)[(x.sum(axis=1) > x.sum(axis=1).mean()).astype(int)]

myNN = NeuroNetwork()
myNN.setLayers([
    FullyConnected(3, 10),
    FunctionLayer(sigmoid()),
    FullyConnected(10, 2),
    FunctionLayer(sigmoid()),
    LossFunctionLayer(SumSquardError()),
])
optim = SGD(myNN, 0.001)
for i in range(100000):
    print(optim.train(x, y).mean())

print(optim.forward(x))
print(y)
import numpy as np
from network import NeuroNetwork


class SGD(object):
    def __init__(self, network:NeuroNetwork, lr:float=0.0001):
        """

        --input parameters-----
        network: network to be trained
        lr: learning rate (default as 0.0001)
        """

        self.network = network
        self.lr = lr
    
    def forward(self, x:np.ndarray) -> np.ndarray:
        """
        calculate the output except loss layer using given x

        --input parameters-----
        x:np.ndarray(samples, in_channels, x, y) : given input
        """

        X = x
        # iterating each leyer to obtain layer-wise output except loss layer
        for i in range(len(self.network.layers)- 1):
                X = self.network.layers[i].forward(X)
        return X

    def forward_with_loss(self, x:np.ndarray, y:np.ndarray):
        """
        calculate the output and loss layer using given x

        --input parameters-----
        x:np.ndarray(samples, in_channels, x, y) : given input
        y:np.ndarray(sample, *output_shape) : expected output

        --outputs--------------
        X:np.ndarray(samples, *output_shape) : network output except loss layer
        loss:np.ndarray(samples, *output_shape) : element-wise loss
        """

        X = self.forward(x)
        # returning final layer output and loss 
        return X, self.network.layers[-1].forward(X, y)
    
    def backward(self, y_:np.ndarray, y:np.ndarray):
        """
        back-propagate using given output by self.forward and given y

        --input parameters-----
        y_:np.ndarray(samples, *output_shape) : given output by self.forward()
        y:np.ndarray(samples, *output_shape) : expected output
        """

        # iterating each leyer to obtain layer-wise derivative
        for i in range(len(self.network.layers) - 1, -1, -1):
            if i == len(self.network.layers) - 1:
                # if it is loss layer
                # it needs given and expected output 
                # to generate original loss derivative
                X = self.network.layers[i].backward(y_, y)
            else:
                X = self.network.layers[i].backward(X)
    
    def step(self):
        """
        update all parameters of NeuroNetwork after self.backward()
        """

        # using Layer.step(lr)
        for i in range(len(self.network.layers)):
            self.network.layers[i].step(self.lr)

    def train(self, x:np.ndarray, y:np.ndarray):
        """
        training api for NeuroNetwork

        --input parameters-----
        x:np.ndarray(samples, in_channels, x, y) : given input
        y:np.ndarray(sample, *output_shape) : expected output

        --outputs--------------
        ret: element-wise loss
        """

        y_, ret = self.forward_with_loss(x, y)
        self.backward(y_, y)
        self.step()
        return ret
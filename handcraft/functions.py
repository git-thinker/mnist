import numpy as np
from abc import abstractmethod, ABC

class PlainFunction(ABC):
    def __init__(self, jacobin:bool=False):
        self.jacobin:bool = jacobin
    
    @abstractmethod
    def forward(self, z:np.ndarray)->np.ndarray:
        pass

    @abstractmethod
    def derivate(self, z:np.ndarray)->np.ndarray:
        pass

class sigmoid(PlainFunction):
    def __init__(self):
        super(sigmoid, self).__init__(False)

    def forward(self, z:np.ndarray)->np.ndarray:
        return 1.0/(1.0 + np.exp(-z))

    def derivate(self, z:np.ndarray)->np.ndarray:
        y = self.forward(z)
        return y*(1-y)

class relu(PlainFunction):
    def __init__(self):
        super(relu, self).__init__(False)

    def forward(self, z:np.ndarray)->np.ndarray:
        return np.maximum(z, 0)
    
    def derivate(self, z:np.ndarray)->np.ndarray:
        return np.heaviside(z, 0.5)

class leakyrelu(PlainFunction):
    def __init__(self):
        super(leakyrelu, self).__init__(False)

    def forward(self, z:np.ndarray)->np.ndarray:
        return np.maximum(z, 0) + np.minimum(z, 0) * 0.2
    
    def derivate(self, z:np.ndarray)->np.ndarray:
        return np.heaviside(z, 0.7 / 1.2) * 1.2 - 0.2


class softmax(PlainFunction):
    # https://zhuanlan.zhihu.com/p/37740860

    def __init__(self):
        super(softmax, self).__init__(True)

    def forward(self, z:np.ndarray)->np.ndarray:
        # a = np.exp(z)
        a = np.exp(z-np.max(z))
        return a / np.sum(a)

    def derivate(self, z:np.ndarray)->np.ndarray:
        """
        da/dz = aE - a.T*a
        """
        a = self.forward(z)
        return np.diag(a.reshape((-1,))) - np.dot(a.reshape((-1, 1)), a.reshape((1, -1)))

class SumSquardError(object):
    def forward(self, y_:np.ndarray, y:np.ndarray)->np.ndarray:
        return (y_ - y) ** 2 / 2

    def derivate(self, y_:np.ndarray, y:np.ndarray)->np.ndarray:
        return (y_ - y)

class CrossEntropyLoss(object):
    # https://zhuanlan.zhihu.com/p/99923080
    def forward(self, y_:np.ndarray, y:np.ndarray)->np.ndarray:
        # (sample, one_hot)
        return - (np.log(y_) * y).sum(axis=1)
    
    def derivate(self, y_:np.ndarray, y:np.ndarray)->np.ndarray:
        return - y / y_



import numpy as np
from abc import abstractmethod
from functions import PlainFunction, ABC


class ShapeError(Exception):
    def __init__(self, progress, correct_shape, wrong_shape):
        self.s = f"Shape not matched in {progress}. {correct_shape} is required, {wrong_shape} received."
    
    def __str__(self):
        return self.s

class Layer(ABC):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.forward(x)

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, d: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def step(self, lr):
        pass

class ReshapeLayer(Layer):
    def __init__(self, in_shape: tuple, out_shape: tuple):
        self.in_shape: tuple = in_shape
        self.out_shape: tuple = out_shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.in_shape is None:
            self.in_shape = x.shape
        if x.shape[1:] != self.in_shape:
            raise ShapeError('ReshapeLayer forward', self.in_shape, x.shape)
        return x.reshape((-1, *self.out_shape))

    def backward(self, d: np.ndarray) -> np.ndarray:
        if d.shape[1:] != self.out_shape:
            raise ShapeError('ReshapeLayer backward', self.out_shape, d.shape)
        return d.reshape((-1, *self.in_shape))

    def step(self, lr):
        pass

class FunctionLayer(Layer):
    def __init__(self, function: PlainFunction):
        self.function: PlainFunction = function

    def forward(self, z: np.ndarray) -> np.ndarray:
        self.z: np.ndarray = z
        return self.function.forward(z)
    
    def backward(self, d: np.ndarray) -> np.ndarray:
        if self.function.jacobin:
            shape = d.shape
            d = d.reshape((1, -1))
            return np.dot(d, self.function.derivate(self.z).T).reshape(shape)
        else:
            return self.function.derivate(self.z) * d
    
    def step(self, lr):
        pass

class LossFunctionLayer(Layer):
    def __init__(self, function:PlainFunction):
        self.function:PlainFunction = function

    def forward(self, y_:np.ndarray, y:np.ndarray) -> np.ndarray:
        self.y_:np.ndarray = y_
        self.y:np.ndarray = y
        return self.function.forward(y_, y)
    
    def backward(self, y_:np.ndarray, y:np.ndarray) -> np.ndarray:
        return self.function.derivate(y_, y)

    def step(self, lr):
        pass

class FullyConnected(Layer):
    """
    w (in_size, out_size)
    b (1, out_size)
    x (1, in_size)
    d (1, out_size)
    z = np.dot(x, w) + b
    dz/dw = x.T
    dl/dw = np.dot(dz/dw, d)
    dz/db = 1
    dl/db = d
    """

    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.w:np.ndarray = np.random.normal(loc=0.0, scale=1.0, size=(self.in_size, self.out_size))
        self.b:np.ndarray = np.random.normal(loc=0.0, scale=1.0, size=(1, self.out_size))
        self.x:np.ndarray = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        if x.shape[1] != self.in_size:
            raise ShapeError('FullyConnected forward', (-1, self.in_size), x.shape)
        return np.dot(x, self.w) + self.b

    def backward(self, d: np.ndarray) -> np.ndarray:
        ret = np.dot(d, self.w.T)
        self.dw = np.dot(self.x.T, d)
        self.db = np.mean(d, axis=0)
        return ret

    def step(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

class Conv2d(Layer):
    """
    input_data(samples, channels, height, width)
    output_data(samples, kernal_num, x, y)
    """

    def __init__(self, in_shape, kernal_size, kernal_num, stride=1, padding='valid'):
        if padding == 'valid':
            padding = 0
        elif padding == 'same':
            padding = (kernal_size - 1) // 2
        elif type(padding) != int:
            raise TypeError(f'padding syntax error. Expected int/valid/same, got{type(padding)}')

        self.in_shape:tuple = in_shape
        # channel, x, y
        self.kernal_size:int = kernal_size
        self.kernal_num:int = kernal_num
        self.stride:int = stride
        self.padding:int = padding
        self.kernal_shape:tuple = (self.kernal_num, self.in_shape[0], self.kernal_size, self.kernal_size)
        self.kernal:np.ndarray = np.random.normal(loc=0.0, scale=1.0, size=self.kernal_shape)
        self.bias:np.ndarray = np.random.normal(loc=0.0, scale=1.0, size=(1,))
        self.x:np.ndarray = None
        self.dkernal:np.ndarray = None
        self.dbias:np.ndarray = None
        # output shape(kernal_num, output_x, output_y)
        self.output_shape:tuple = \
            (self.kernal_num, 
                *self.conv_output_shape
                (
                    self.in_shape[1:], 
                    self.kernal_size, 
                    self.stride, 
                    self.padding,
                )
            )
                    

    def forward(self, x:np.ndarray):
        # checking input size
        if x.shape[-3:] != self.in_shape:
            raise ShapeError('CNN forward', self.in_shape, x.shape)

        # setting empty output
        output = np.zeros((x.shape[0], *self.output_shape))
        
        # padding
        # no padding on dimansion of samples & channels
        x = np.pad(x, 
            (
                (0, 0), 
                (0, 0), 
                (self.padding, self.padding),
                (self.padding, self.padding),
            )
        )

        # saving x for backward
        self.x = x
        
        for i in range(0, self.in_shape[1], self.stride):
            for j in range(0, self.in_shape[2], self.stride):
                x_sliced = x[:, :, i:i+self.kernal_size, j:j+self.kernal_size]
                output[:, :, i, j] = self.conv_one_step(x_sliced)
        
        return output

    def backward(self, d):
        # https://zhuanlan.zhihu.com/p/81675803
        # loss shall have the same shape as output
        # loss(samples, kernal_num, x, y)
        
        # checking loss shape
        if d.shape[-3:] != self.output_shape:
            raise ShapeError('CNN backward', self.output_shape, d.shape)
        self.dkernal = np.zeros(self.kernal_shape)
        for i in range(0, self.kernal_shape[0]):
            for j in range(0, self.kernal_shape[2], self.stride):
                for k in range(0, self.kernal_shape[3], self.stride):
                    self.dkernal[i, :, j, k] = (self.x[:, :, j:j+self.in_shape[1], k:k+self.in_shape[2]] * d[:,i,:,:][:, np.newaxis, : , : ]).sum(axis=(0,2,3))

        # zero padding for loss
        d = np.pad(d,
            (
                (0, 0), 
                (0, 0), 
                (self.padding, self.padding),
                (self.padding, self.padding),
            )
        )

        backward_kernal = self.kernal.transpose(0, 1, 3, 2)

        # set up 
        ret = np.zeros((self.x.shape[0], *self.in_shape))
        for i in range(0, self.x.shape[0]):
            for j in range(0, self.in_shape[1], self.stride):
                for k in range(0, self.in_shape[2], self.stride):
                    # ret[i, :, j, k] = (d[:, i, j:j+self.kernal_size, k:k+self.kernal_size] * backward_kernal).sum(axis=(0,2,3))
                    ret[i, :, j, k] = (d[i, :, j:j+self.kernal_size, k:k+self.kernal_size][:, np.newaxis, :,:] * backward_kernal).sum(axis=(0,2,3))
        return ret

    def step(self, lr):
        self.kernal -= self.dkernal * lr

    def conv_one_step(self, x_sliced:np.ndarray):
        """
        input:(samples, channels, x, y) sliced
        kernal:(kernal_num, channels, x, y)
        output(samples, kernal_num)
        Hadamard product on x, y and broadcast on channel
        """
        return np.einsum('ijxy,kjxy->ik', x_sliced, self.kernal)

    def conv_output_size(self, input_size, filter_size, stride=1, pad=0):
        return (input_size + 2*pad - filter_size) // stride + 1

    def conv_output_shape(self, input_shape, filter_size, stride=1, pad=0):
        return self.conv_output_size(input_shape[0], filter_size, stride, pad), self.conv_output_size(input_shape[1], filter_size, stride, pad)

class MaxPooling(Layer):
    def __init__(self, input_size, kernal_size):
        """
        input_size:tuple (x, y)
        kernal_size:int kernal_size on dimansion
        """

        self.input_size = input_size
        self.kernal_size = kernal_size

    pass

class MeanPooling(Layer):
    def __init__(self, input_size, kernal_size):
        """
        input_size:tuple (x, y)
        kernal_size:int kernal_size on dimansion
        """

        self.input_size = input_size
        self.kernal_size = kernal_size
    
    def forward(self, x:np.ndarray):
        output = np.zeros((*x.shape[:2], *self.pooling_output_shape(self.input_size, self.kernal_size)))
        for i in range(0, self.input_size[0] // self.kernal_size):
            for j in range(0, self.input_size[1] // self.kernal_size):
                output[:, :, i, j] = x[:, :, i*self.kernal_size:(i+1)*self.kernal_size, j:(j+1)*self.kernal_size].mean(axis=(2, 3))

        return output        

    def backward(self, d:np.ndarray):
        output = np.zeros((*d.shape[:2], *self.input_size))
        for i in range(0, d.shape[2]):
            for j in range(0, d.shape[3]):
                output[:, :, i*self.kernal_size:(i+1)*self.kernal_size, j:(j+1)*self.kernal_size] = (d[:, :, i, j] / self.kernal_size ** 2)[:, : ,np.newaxis, np.newaxis]
        return output        

    def step(self, lr):
        pass

    def pooling_output_shape(self, input_size, kernal_size):
        return (input_size[0] // kernal_size, input_size[1] // kernal_size)
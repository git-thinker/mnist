from layers import *
from functions import *
from network import *
from optimiser import *

import cv2
import pickle

dataset = pickle.load(open('mnist.pkl','rb'),encoding='iso-8859-1')

# using set 0
# x is 
x = dataset[0][0][:500].reshape((-1, 1, 28, 28))
y = dataset[0][1][:500]
y_onehot = np.eye(10)[y]

myNN = NeuroNetwork()
myNN.setLayers(
    [
        Conv2d(
            in_shape=(1, 28, 28), 
            kernal_size=5,
            kernal_num=32,
            stride=1,
            padding='same',
        ),
        FunctionLayer(sigmoid()),
        MeanPooling(
            input_size=(28, 28),
            kernal_size=2,
        ),
        Conv2d(
            in_shape=(32, 14, 14),
            kernal_size=5,
            kernal_num=64,
            stride=1,
            padding='same',
        ),
        FunctionLayer(sigmoid()),
        MeanPooling(
            input_size=(14, 14),
            kernal_size=2,
        ),
        ReshapeLayer(
            in_shape=(64, 7, 7),
            out_shape=(3136,)
        ),
        FullyConnected(
            in_size=3136,
            out_size=1024,
        ),
        FunctionLayer(sigmoid()),
        FullyConnected(
            in_size=1024,
            out_size=10,
        ),
        FunctionLayer(softmax()),
        LossFunctionLayer(CrossEntropyLoss())
    ]
)

optim = SGD(myNN, 0.0001)
loss_list = []
valid_list = []
for i in range(100000):
    loss = optim.train(x[i:i+100], y_onehot[i:i+100]).mean()
    loss_list.append(loss)
    print(loss)
    test = optim.forward(x[-20:]).argmax(axis=1)
    print(test)
    print(y[-20:])
    valid_list.append((test == y[-20:]).mean())
    pickle.dump((i, loss_list, valid_list, myNN), open('cnn.pkl','wb'))

    if i == 0 or (valid_list[-1] > valid_list[-2]):
        pickle.dump((i, loss, valid_list, myNN), open('cnn_best.pkl', 'wb'))





'''
a = cv2.imread('1.png').T[np.newaxis,:,:,:]
c = Conv2d(a.shape[1:], 3, 3, 1, 'same')
c.kernal = np.array([[[[1,1,1],[0,0,0],[-1,-1,-1]],[[1,1,1],[0,0,0],[-1,-1,-1]],[[1,1,1],[0,0,0],[-1,-1,-1]]]])
c.kernal = np.random.randint(-1,1,size=(3,3,3,3))
y = c.forward(a)
ll = LossFunctionLayer(SumSquardError())
d = ll.backward(a, y)
dd = c.backward(d)
'''
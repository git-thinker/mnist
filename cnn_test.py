from layers import *
from functions import *
from network import *
from optimiser import *

import pickle
import argparse

parser = argparse.ArgumentParser(description='This is a MNIST Recogniser based on Numpy-CNN')
parser.add_argument('--epochs', type=int, help='trainning epochs', default=10)
parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
parser.add_argument('--batch_size', type=int, help='batch size', default=100)
parser.add_argument('--pre_trained', type=str, help='optional pre-train model path')
args = parser.parse_args()

epochs = args.epochs
lr = args.lr
batch_size = args.batch_size

if args.pre_trained is not None:
    data = pickle.load(open(args.pre_trained, 'rb'))
    epoch_start = data['epoch_start']
    batch_start = data['batch_start']
    loss_list = data['loss_list']
    valid_list = data['valid_list']
    myNN = data['NN']

else:
    batch_start = 0
    epoch_start = 0
    i = 0
    loss_list = []
    valid_list = []

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

dataset = pickle.load(open('mnist.pkl','rb'),encoding='iso-8859-1')

# dividing dataset

x = np.concatenate((dataset[0][0], dataset[1][0], dataset[2][0][:-1100,:]),axis=0).reshape((-1, 1, 28, 28))
y = np.concatenate((dataset[0][1], dataset[1][1], dataset[2][1][:-1100]),axis=0)
y_onehot = np.eye(10)[y]

x_valid = dataset[2][0][-1100:-1000,:].reshape((-1, 1, 28, 28))
y_valid = dataset[2][1][-1100:-1000]
y_valid_onehot = np.eye(10)[y]

x_test = dataset[2][0][-1000:,:].reshape((-1, 1, 28, 28))
y_test = dataset[2][1][-1000:]
y_test_onehot = np.eye(10)[y_test]


optim = SGD(myNN, lr)


for epoch in range(epoch_start, epochs):
    if epoch == epoch:
        start = batch_start
    else:
        start = 0

    for i in range(start, len(x), batch_size):
        print('epoch %d / %d, progress %f %%' % (epoch, epochs, i / len(x) * 100))

        loss = optim.train(x[i:i+batch_size], y_onehot[i:i+batch_size]).mean()
        valid = (optim.forward(x_valid).argmax(axis=1) == y_valid).mean()

        print('-------------')
        print('loss: ', loss)
        print('valid: ', valid)
        print('\n')

        loss_list.append(loss)
        valid_list.append(valid)

        dumped = {
            'epoch_start': epoch,
            'batch_start': i,
            'loss_list': loss_list,
            'valid_list': valid_list,
            'NN': myNN
        }


        pickle.dump(dumped, open('cnn.pkl','wb'))

        if i == 0 or (valid_list[-1] == max(valid_list)):
            pickle.dump(dumped, open('cnn_best.pkl', 'wb'))

test = optim.forward(x_test).argmax(axis=1)
print(test)
print(y_test)
print((test == y_test).mean())

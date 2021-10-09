import torch
import matplotlib.pyplot as plt

class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=3, out_features=10)
        self.fc2 = torch.nn.Linear(in_features=10, out_features=2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.sigmoid(x)
        x = self.fc2(x)
        x = torch.nn.functional.sigmoid(x)
        return x

nn = NN()
x = torch.rand(size=(1000, 3))
y = torch.eye(2)[(x.sum(axis=1) > x.sum(axis=1).mean()).long()]

loss_func = torch.nn.MSELoss()
optimiser = torch.optim.SGD(nn.parameters(), lr = 0.5)

for i in range(10000):
    output = nn(x)           # cnn output
    loss = loss_func(output, y)   # cross entropy loss
    optimiser.zero_grad()           # clear gradients for this training step
    loss.backward()                 # backpropagation, compute gradients
    optimiser.step()                # apply gradients
    print(loss.sum())
print(output, y)
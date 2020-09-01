import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os

class RegressionModel(torch.nn.Module):
    def __init__(self):
        super(RegressionModel,self).__init__()
        self.W1 = torch.randn(784,10,requires_grad=True)
        self.b1 = torch.randn(1,10,requires_grad=True)

    def logits(self,x):
        return x @ self.W1 + self.b1

    def f(self, x):
        return torch.nn.functional.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

def getData():
    mnist_train = torchvision.datasets.MNIST('./data', train = True, download = True)
    x_train = mnist_train.data.reshape(-1,784).float()
    y_train = torch.zeros((mnist_train.targets.shape[0],10))
    y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1.

    mnist_test = torchvision.datasets.MNIST('./data', train=False,download=True)
    x_test = mnist_test.data.reshape(-1,784).float()
    y_test = torch.zeros((mnist_test.targets.shape[0],10))
    y_test[torch.arange(mnist_test.targets.shape[0]),mnist_test.targets] = 1

    return x_train,y_train,x_test,y_test

def train(x,y):
    checkpoint = 100
    model = RegressionModel()
    if os.path.isfile('progressMNIST.pt'):
        model.W1,model.b1 = torch.load('progressMNIST.pt')
    model.eval()
    optimizer = torch.optim.SGD([model.b1,model.W1],float('1.0e-5'))
    for i in range(checkpoint):
        for epoch in range(10):
            model.loss(x,y).backward()
            optimizer.step()
            optimizer.zero_grad()
        print("loss = %s  >-->-->--> %s/%s" %
              (model.loss(x, y), i, checkpoint))
        torch.save([model.W1,model.b1],'progressMNIST.pt')
    return model


def main():
    x_train, y_train, x_test, y_test = getData()
    while True:
        model = train(x_train,y_train)
        accuracy = model.accuracy(x_test,y_test)
        print('Final accuracy: %s'%accuracy)
        if accuracy>0.9:
            break

    found = [False, False, False, False, False, False, False, False, False, False]
    count = 0
    while False in found:
        for i in range(10):
            print(y_test[i, count].item())
            if y_test[i, count].item() == 1.:
                print(y_test[count, i].item())
                print(count)
                if not found[i]:
                    plt.imshow(x_train[count, :].reshape(28, 28))
                    plt.imsave('x_train_%s.png' % i, x_train[count, :].reshape(28, 28))
                    found[i] = True
                else:
                    break
        count += 1


if __name__ == '__main__':
    main()
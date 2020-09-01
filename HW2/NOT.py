import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import interpolate

x_train = torch.tensor([[0.], [1.]])
y_train = torch.tensor([[1.], [0.]])

class RegressionModel(torch.nn.Module):
    def __init__(self):
        super(RegressionModel,self).__init__()
        self.W = torch.tensor([[0.0]],requires_grad=True)
        self.b = torch.tensor([[0.0]],requires_grad=True)

    def logits(self,x):
        return torch.sigmoid(x @ self.W + self.b)

    def loss(self,x,y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x),y)

def train(x_given, y_true):
    checkpoint = 100
    model = RegressionModel()
    if os.path.isfile('progressNOT.pt'):
        model.W,model.b = torch.load('progressNOT.pt')
    model.eval()
    optimizer = torch.optim.SGD([model.b,model.W],float('1.0e-6'))
    for i in range(checkpoint):
        for epoch in range(100000):
            model.loss(x_given, y_true).backward()
            optimizer.step()
            optimizer.zero_grad()
        print("W = %s, b = %s, loss = %s  >-->-->--> %s/%s " % (model.W, model.b, model.loss(x_given, y_true), i, checkpoint))
        torch.save([model.W,model.b],'progressNOT.pt')
    return model

def draw_res(x_train,y_train,model):
    plt.plot(x_train,y_train,'o')
    plt.xlabel('x')
    plt.ylabel('y')
    x_new = np.linspace(0, 1, 360)
    plot_path = interpolate.make_interp_spline(x_new, model.logits(torch.tensor(x_new).reshape(-1, 1).float()).detach())
    y_new = plot_path(x_new)
    plt.plot(x_new,y_new)
    plt.show()

def main():
    model = train(x_train,y_train)
    draw_res(x_train,y_train,model)

if __name__ == '__main__':
    main()
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

x_train = torch.tensor([[0.,0.],
                        [1.,0.],
                        [0.,1.],
                        [1.,1.]])
y_train = torch.tensor([[1.],
                        [1.],
                        [1.],
                        [0.]])

class RegressionModel(torch.nn.Module):
    def __init__(self):
        super(RegressionModel,self).__init__()
        self.W = torch.tensor([[0.0],[0.0]],requires_grad=True)
        self.b = torch.tensor([[0.0]],requires_grad=True)

    def logits(self,x):
        return torch.sigmoid(x@self.W+self.b)

    def loss(self,x,y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x),y)

def train(x_given, y_true):
    checkpoint = 100
    model = RegressionModel()
    if os.path.isfile('progressNAND.pt'):
        model.W,model.b = torch.load('progressNAND.pt')
    model.eval()
    optimizer = torch.optim.SGD([model.b,model.W],float('1.0e-6'))
    for i in range(checkpoint):
        for epoch in range(100000):
            model.loss(x_given,y_true).backward()
            optimizer.step()
            optimizer.zero_grad()
        print("W = %s, b = %s, loss = %s  >-->-->--> %s/%s " % (model.W, model.b, model.loss(x_given, y_true), i, checkpoint))
        torch.save([model.W,model.b],'progressNAND.pt')
    return model

def plot_graph(x,y,z,model):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    x_ax = np.arange(0, 1, 0.01)
    y_ax = np.arange(0, 1, 0.01)
    X,Y = np.meshgrid(x_ax,y_ax)
    z_calc = torch.transpose(torch.tensor([np.ravel(X),np.ravel(Y)]),0,1).float()
    z_ax = np.array(model.logits(z_calc).detach())
    Z=z_ax.reshape(X.shape)

    ax.scatter(x,y,z,marker='o',color='r')
    ax.plot_wireframe(X,Y,Z,rstride=10,cstride=10)

    plt.show()

def main():
    model = train(x_train,y_train)
    plot_graph(x_train[:,0],x_train[:,1],y_train,model)

if __name__ == '__main__':
    main()
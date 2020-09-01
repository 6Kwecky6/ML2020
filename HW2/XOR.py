import torch
import matplotlib.pyplot as plt
import numpy as np
import os

x_train = torch.tensor([[0.,0.],
                        [1.,0.],
                        [0.,1.],
                        [1.,1.]])
y_train = torch.tensor([[0.],
                        [1.],
                        [1.],
                        [0.]])

class RegressionModel(torch.nn.Module):
    def __init__(self):
        super(RegressionModel,self).__init__()
        self.W1 = torch.tensor([[1.,-1.],
                               [1.,-1.]], requires_grad=True)
        self.b1 = torch.tensor([0.,0.],requires_grad=True)

        self.W2 = torch.tensor([[0.],[0.]],requires_grad=True)
        self.b2 = torch.tensor([[0.]], requires_grad=True)

    def logits1(self, x):
        return torch.sigmoid(x @ self.W1 + self.b1)

    def logits2(self, x):
        return torch.sigmoid(x @ self.W2 + self.b2)

    def logit(self,x):
        return self.logits2(self.logits1(x))

    def loss(self,x,y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logit(x),y)

def train(x_given,y_true):
    checkpoint = 100
    model = RegressionModel()
    if os.path.isfile('progressXOR.pt'):
        model.W1,model.b1,model.W2,model.b2 = torch.load('progressXOR.pt')
    model.eval()
    optimizer = torch.optim.SGD([model.b1,model.W1,model.b2,model.W2],float('1.0e-4'))
    for i in range(checkpoint):
        for epoch in range(100):
            model.loss(x_given,y_true).backward()
            optimizer.step()
            optimizer.zero_grad()
        print("W1 = %s, b1 = %s\nW2 = %s, b2 = %s\nloss = %s  >-->-->--> %s/%s"%
              (model.W1,model.b1,model.W2,model.b2,model.loss(x_given,y_true),i,checkpoint))
        torch.save([model.W1,model.b1,model.W2,model.b2],'progressXOR.pt')
    return model

def plot_graph(x,y,z,model):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    x_ax = np.arange(0, 1, 0.01)
    y_ax = np.arange(0, 1, 0.01)
    X,Y = np.meshgrid(x_ax,y_ax)
    z_calc = torch.transpose(torch.tensor([np.ravel(X),np.ravel(Y)]),0,1).float()
    z_ax = np.array(model.logit(z_calc).detach())
    Z=z_ax.reshape(X.shape)

    ax.scatter(x,y,z,marker='o',color='r')
    ax.plot_wireframe(X,Y,Z,rstride=10,cstride=10)

    plt.show()

def main():
    model = train(x_train,y_train)
    plot_graph(x_train[:,0],x_train[:,1],y_train,model)

if __name__ == '__main__':
    main()
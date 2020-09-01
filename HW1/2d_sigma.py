import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import csv


# Problem: Finner et ikke-optimalt bunnpunkt for loss
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[-6.0]], requires_grad=True)

    def forward(self,x):
        return

    # Predictor
    def f(self, x):
        y = x @ self.W + self.b
        return 20.**torch.sigmoid(y)+31.  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


# Function to read from file
# Returns a list of training data
def readfile(file_name):
    with open(file_name,newline="") as f:
        x_file=[]
        y_file=[]
        read = csv.reader(f)
        next(read)
        for row in read:
            x_file.append(float(row[0]))
            y_file.append(float(row[1]))
        x_train=torch.tensor(x_file).reshape(-1,1)
        y_train=torch.tensor(y_file).reshape(-1,1)
        print('x_train:')
        print(x_train)
        print('y_train:')
        print(y_train)

    return [x_train,y_train]

# Function to train the algorithm
# Returns an optimised weight and base
def train(x_train, y_train):
    checkpoints=100
    model = LinearRegressionModel()
    model.W, model.b= torch.load('trainingSigma.pt')
    model.eval()
    # Optimize: adjust W and b to minimize loss using stochastic gradient descent
    optimizer = torch.optim.SGD([model.b, model.W], 0.00000001)
    for i in range(checkpoints):
        for epoch in range(1000):
            model.loss(x_train, y_train).backward()  # Compute loss gradients
            optimizer.step()  # Perform optimization by adjusting W and b,
            optimizer.zero_grad()  # Clear gradients for next step
        print("W = %s, b = %s, loss = %s  >-->-->--> %s/%s " % (model.W, model.b, model.loss(x_train, y_train),i,checkpoints))
        torch.save([model.W,model.b], 'trainingSigma.pt')
    return model


# Function to create a visual model of the results
def graph_res(x_train,y_train, model):
    # Visualize result
    plt.plot(x_train, y_train, 'o')
    plt.xlabel('x')
    plt.ylabel('y')
    x_new = np.linspace(0,1800,360)
    plot_path = interpolate.make_interp_spline(x_new,model.f(torch.tensor(x_new).reshape(-1,1).float()).detach())
    y_new = plot_path(x_new)
    print(x_new)
    plt.plot(x_new, y_new)
    plt.show()


def main():
    res=readfile('day_head_circumference.csv')
    x_train = res[0]
    y_train = res[1]
    model=train(x_train,y_train)
    graph_res(x_train,y_train,model)


if __name__ == '__main__':
    main()

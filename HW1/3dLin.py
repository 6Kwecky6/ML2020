import torch
import csv
import matplotlib.pyplot as plt
import numpy as np

class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # Model variables
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[-6.0]], requires_grad=True)

    def forward(self,x):
        return

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))

# Function to read from file
# Returns a list of training data
def readfile(file_name):
    with open(file_name,newline="") as f:
        day=[]
        length=[]
        weight=[]
        read = csv.reader(f)
        next(read)
        for row in read:
            day.append(float(row[0]))
            length.append(float(row[1]))
            weight.append(float(row[2]))
        print(length)
        x_train=torch.transpose(torch.tensor([day,length]),0,1)
        y_train=torch.tensor(weight).reshape(-1,1)
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

    model.W, model.b= torch.load('trainingRes3d.pt')
    model.eval()

    # Optimize: adjust W and b to minimize loss using stochastic gradient descent
    optimizer = torch.optim.SGD([model.b, model.W], 0.0000001)
    for i in range(checkpoints):
        for epoch in range(100):
            model.loss(x_train, y_train).backward()  # Compute loss gradients
            optimizer.step()  # Perform optimization by adjusting W and b,
            optimizer.zero_grad()  # Clear gradients for next step
        print("W = %s, b = %s, loss = %s  >-->-->--> %s/%s " % (model.W, model.b, model.loss(x_train, y_train),i,checkpoints))
        torch.save([model.W,model.b], 'trainingRes3d.pt')
    return model

#Function to create a graph
def graph_res(x,y,z,model):
    # Messy code
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    x_ax = np.arange(0,2000,100)
    y_ax = np.arange(0,150,5)
    X,Y = np.meshgrid(x_ax,y_ax)
    z_calc=torch.transpose(torch.tensor([np.ravel(X),np.ravel(Y)]),0,1).float()
    z_ax = np.array(model.f(z_calc).detach())
    Z=z_ax.reshape(X.shape)

    ax.scatter(x,y,z,marker='o') #Creates points
    ax.plot_surface(X,Y,Z, cmap=plt.cm.YlGnBu_r) #Creates surface

    plt.show() #Creates magic

def main():
    res = readfile('day_length_weight2.csv')
    x_train = res[0]
    y_train = res[1]
    model=train(x_train,y_train)
    xy=x_train.transpose(0,1)
    graph_res(xy[0],xy[1],y_train.transpose(0,1),model)


if __name__ == '__main__':
    main()

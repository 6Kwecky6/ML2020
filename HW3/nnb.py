import torch
import torch.nn as nn
import torchvision
import os

torch.cuda.set_device(0)
# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

# Normalization of inputs
mean = x_train.mean()
std = x_train.std()
x_train = ((x_train - mean) / std)
x_test = ((x_test - mean) / std)

# Divide training data into batches to speed up optimization
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)


class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        # Model layers (includes initialized model variables):
        self.conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lin = nn.Linear(64*7*7,1024)
        self.dense = nn.Linear(1024, 10)



    def logits(self, x):
        x = x.cuda()
        x = self.conv(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.lin(x.reshape(-1, 64 * 7 * 7))
        return self.dense(x)

    # Predictor
    def forward(self, x):
        x = x.cuda()
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        x = x.cuda()
        y = y.cuda()
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        x = x.cuda()
        y = y.cuda()
        return torch.mean(torch.eq(self.forward(x).argmax(1), y.argmax(1)).float())


model = ConvolutionalNeuralNetworkModel()
x_train.cuda()
x_test.cuda()
y_train.cuda()
y_test.cuda()
model.cuda()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), float('1.0e-5'))

if os.path.isfile('nnb_state.pt'):
    state = torch.load('nnb_state.pt')
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])

for epoch in range(20):
    for batch in range(len(x_train_batches)):
        model.loss(x_train_batches[batch].cuda(), y_train_batches[batch].cuda()).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step
    print("accuracy = %s" % model.accuracy(x_test, y_test))
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict':optimizer.state_dict()
},'nnb_state.pt')

# Accuracy: 0.9851
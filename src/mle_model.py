# create our dataset
X = np.array([dataset.features[0], dataset.features[2]]).T
y = dataset.goals

import torch
from torch.autograd import Variable

torch.manual_seed(12)


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return torch.exp(out)


def my_loss(output, target):
    loss = -target * torch.log(output) + output
    return torch.mean(loss)


inputDim = 2  # takes variable 'x'
outputDim = 1  # takes variable 'y'
learningRate = 0.001
epochs = 100

model = linearRegression(inputDim, outputDim)
model.double()

criterion = my_loss  # torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

for epoch in range(1000):
    # Converting inputs and labels to Variable

    inputs = Variable(torch.from_numpy(X))
    labels = Variable(torch.from_numpy(y))

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels)
    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()
    if epoch % 100 == 0:
        print("epoch {}, loss {}".format(epoch, loss.item()))

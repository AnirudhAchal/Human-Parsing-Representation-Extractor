import torch
import torch.nn.functional as F
import numpy as np


# Const Variables
N_FEATURES = 5
EPOCHS = 20
LEARNING_RATE = 0.01
VERBOSE = 10


# One Hot Encoding
h = [1, 0, 0, 0, 0]
e = [0, 1, 0, 0, 0]
l = [0, 0, 1, 0, 0]
i = [0, 0, 0, 1, 0]
o = [0, 0, 0, 0, 1]


class Net(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_features, 8)
        self.fc2 = torch.nn.Linear(8, 5)
        self.fc3 = torch.nn.Linear(5, 5)

    def forward(self, X):
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))
        X = torch.relu(self.fc3(X))

        return X

    def calculate_accuracy(self, X, y):

        # Number of training examples 
        m = y.shape[0]

        # Declaring tensors for torch.where()
        one = torch.ones([m, 1])
        zero = torch.zeros([m, 1])

        # Forward Pass
        y_pred = self(X)
        y_pred = torch.argmax(y_pred, dim = 1)

        return torch.sum(y_pred == y).item() / m * 100

    def train(self, X, y, EPOCHS, learning_rate, verbose):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)

        for epoch in range(EPOCHS):
            # Forward Pass
            y_pred = self(X)
            accuracy = self.calculate_accuracy(X, y)
            loss = criterion(y_pred, y)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % (EPOCHS // verbose) == 0:
                print(f"Epoch : {epoch} | loss : {round(loss.item(), 5)} | accuracy : {round(accuracy, 2)}")

        print(f"Epoch : {EPOCHS} | loss : {round(loss.item(), 5)} | accuracy : {round(accuracy, 2)}") 
        

if __name__ == "__main__":
    # Import Dataset
    data_X = np.array([h, i, h, e, l, l]).astype('float32')
    data_Y = np.array([3, 0, 2, 3, 2, 4]).ravel()

    X = torch.from_numpy(data_X)
    y = torch.from_numpy(data_Y)

    # Model Instance
    torch.manual_seed(4)
    net = Net(N_FEATURES)
    net.train(X, y, EPOCHS, LEARNING_RATE, VERBOSE)





import torch
import numpy as np

# Const Variables
FILE_NAME = "diabetes.csv"
N_FEATURES = 0
EPOCHS = 0
LEARNING_RATE = 0.1

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 8)
        self.fc2 = torch.nn.Linear(8, 1)

    def forward(self, X):
        X = torch.sigmoid(self.fc1(X))
        X = torch.sigmoid(self.fc2(X))
        return X

    def calculate_accuracy(self, X, y):

        # Number of training examples 
        m = y.shape[0]

        # Declaring tensors for torch.where()
        ones = torch.ones([m, 1])
        zero = torch.zeros([m, 1])

        # Forward Pass
        y_pred = self(X)
        y_pred = torch.where(y_pred >= 0.5, one, zero)

        return torch.sum(y_pred == y).item() / m * 100

    def train(self, X, y, EPOCHS, learning_rate, verbose):
        criterion = torch.nn.BCELoss()
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
                print(f"Epoch : {epoch} | loss : {loss} | accuracy : {accuracy}")

        print(f"Epoch : {EPOCHS} | loss : {loss} | accuracy : {accuracy}") 
        

# Import Dataset
data = np.loadtxt(FILE_NAME, delimiter = ',', dtype = np.float)

X = torch.from_numpy(data[:, 0 : -1])
y = torch.from_numpy(data[:, -1])

print(X.shape, y.shape)





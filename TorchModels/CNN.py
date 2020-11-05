import torch
import torch.nn as nn
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# Const Variables
EPOCHS = 1000
LEARNING_RATE = 0.001
VERBOSE = 10


class TrainDataset(Dataset):
    def __init__(self):
        # Import Dataset
        train_dataset = h5py.File('../Datasets/TrainCatsVsNoncats.h5', "r")
        
        train_dataset_X = np.array(train_dataset["train_set_x"][:])
        train_dataset_y = np.array(train_dataset["train_set_y"][:])

        self.len = train_dataset_X.shape[0]
        self.x_data = torch.from_numpy(train_dataset_X).reshape(-1, 3, 64, 64).float() / 256
        self.y_data = torch.from_numpy(train_dataset_y).reshape(-1, 1).float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class TestDataset(Dataset):
    def __init__(self):
        # Import Dataset
        test_dataset = h5py.File('../Datasets/TestCatsVsNoncats.h5', "r")
        
        test_dataset_X = np.array(test_dataset["test_set_x"][:])
        test_dataset_y = np.array(test_dataset["test_set_y"][:])

        self.len = test_dataset_X.shape[0]
        self.x_data = torch.from_numpy(test_dataset_X).reshape(-1, 3, 64, 64).float() / 256
        self.y_data = torch.from_numpy(test_dataset_y).reshape(-1, 1).float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
        

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=10)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=10)

        self.mp = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(1296, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, X):
        dataset_size = X.shape[0]

        # First Convolutional Layer
        X = self.mp(self.conv1(X))
        X = torch.relu(X)

        # Second Convolutional Layer
        X = self.mp(self.conv2(X))
        X = torch.relu(X)
        
        # Flattening
        X = X.view(dataset_size, -1)

        # First Fully Connected Layer
        X = self.fc1(X)
        X = torch.relu(X)

        # Second Fully Connected Layer
        X = self.fc2(X)
        X = torch.sigmoid(X)

        return X

    def calculate_accuracy(self, data_loader):

        correct_count = 0
        total_count = 0

        for data in data_loader:
            # Unpack Data
            X, y = data

            # Number of training examples 
            m = y.shape[0]

            # Declaring tensors for torch.where()
            one = torch.ones([m, 1])
            zero = torch.zeros([m, 1])

            # Forward Pass
            y_pred = self(X)
            y_pred = torch.where(y_pred >= 0.5, one, zero)

            correct_count += torch.sum(y_pred == y).item()
            total_count += m

        print(f"Correct : {correct_count} | Total Count : {total_count}")

        return round(correct_count / total_count * 100, 2)

    def train(self, data_loader, EPOCHS, learning_rate, verbose):
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)

        for epoch in range(EPOCHS):
            for data in data_loader:
                # Unpack data
                X, y = data

                # Forward Pass
                y_pred = self(X)
                loss = criterion(y_pred, y)

                # Backward Pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % (EPOCHS // verbose) == 0:
                print(f"Epoch : {epoch} | loss : {round(loss.item(), 5)}")

        print(f"Epoch : {EPOCHS} | loss : {round(loss.item(), 5)}") 

        
if __name__ == '__main__':
    # Data Loader
    train_data = TrainDataset()
    train_loader = DataLoader(dataset=train_data, batch_size=209, shuffle=True)

    test_data = TestDataset()
    test_loader = DataLoader(dataset=test_data, batch_size=50, shuffle=True)

    # Model Instance
    torch.manual_seed(0)
    net = Net()
    net.train(train_loader, EPOCHS, LEARNING_RATE, VERBOSE)

    train_accuracy = net.calculate_accuracy(train_loader)
    print(f"\nTrain accuracy after {EPOCHS} epochs : {round(train_accuracy, 2)}")

    test_accuracy = net.calculate_accuracy(test_loader)
    print(f"\nTest accuracy after {EPOCHS} epochs : {round(test_accuracy, 2)}\n")





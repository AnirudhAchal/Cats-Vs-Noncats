import os
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# Const Variables
EPOCHS = 50
LEARNING_RATE = 0.001
VERBOSE = 10


def import_data():
    TrainFolder = '../Datasets/TrainImages'
    TestFolder = '../Datasets/TestImages'

    train_dataset_X = []
    train_dataset_y = []
    test_dataset_X = []
    test_dataset_y = []

    for filename in os.listdir(os.path.join(TrainFolder, 'Cats')):
        img = cv2.imread(os.path.join(TrainFolder, 'Cats/' + filename))
        train_dataset_X.append(img)
        train_dataset_y.append(1)

    for filename in os.listdir(os.path.join(TrainFolder, 'Noncats')):
        img = cv2.imread(os.path.join(TrainFolder, 'Noncats/' + filename))
        train_dataset_X.append(img)
        train_dataset_y.append(0)

    for filename in os.listdir(os.path.join(TestFolder, 'Cats')):
        img = cv2.imread(os.path.join(TestFolder, 'Cats/' + filename))
        test_dataset_X.append(img)
        test_dataset_y.append(1)

    for filename in os.listdir(os.path.join(TestFolder, 'Noncats')):
        img = cv2.imread(os.path.join(TestFolder, 'Noncats/' + filename))
        test_dataset_X.append(img)
        test_dataset_y.append(0)

    train_dataset_X = np.array(train_dataset_X).reshape(-1, 3, 64, 64)
    train_dataset_y = np.array(train_dataset_y).reshape(-1, 1)
    test_dataset_X = np.array(test_dataset_X).reshape(-1, 3, 64, 64)
    test_dataset_y = np.array(test_dataset_y).reshape(-1, 1)
    
    return train_dataset_X, train_dataset_y, test_dataset_X, test_dataset_y


class TrainDataset(Dataset):
    def __init__(self):
        # Import Dataset
        train_dataset_X, train_dataset_y, _, _ = import_data()

        self.len = train_dataset_X.shape[0]
        self.x_data = torch.from_numpy(train_dataset_X).float() / 256
        self.y_data = torch.from_numpy(train_dataset_y).float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class TestDataset(Dataset):
    def __init__(self):
        # Import Dataset
        _, _, test_dataset_X, test_dataset_y = import_data()

        self.len = test_dataset_X.shape[0]
        self.x_data = torch.from_numpy(test_dataset_X).float() / 256
        self.y_data = torch.from_numpy(test_dataset_y).float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def len(self):
        return self.len
        

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=8)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=8)

        self.mp = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(12800, 64)
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
    train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True)

    test_data = TestDataset()
    test_loader = DataLoader(dataset=test_data, batch_size=256, shuffle=True)

    # Model Instance
    torch.manual_seed(1)
    net = Net()
    net.train(train_loader, EPOCHS, LEARNING_RATE, VERBOSE)

    train_accuracy = net.calculate_accuracy(train_loader)
    print(f"\nTrain accuracy after {EPOCHS} epochs : {round(train_accuracy, 2)}")

    test_accuracy = net.calculate_accuracy(test_loader)
    print(f"\nTest accuracy after {EPOCHS} epochs : {round(test_accuracy, 2)}\n")



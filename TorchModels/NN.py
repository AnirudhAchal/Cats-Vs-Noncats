import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# Const Variables
N_FEATURES = 64*64*3
EPOCHS = 50
LEARNING_RATE = 0.001
VERBOSE = 10


class TrainDataset(Dataset):
    def __init__(self):
        # Import Dataset
        train_dataset = h5py.File('../Datasets/TrainCatsVsNoncats.h5', "r")
        
        train_dataset_X = np.array(train_dataset["train_set_x"][:])
        train_dataset_y = np.array(train_dataset["train_set_y"][:])

        self.len = train_dataset_X.shape[0]
        self.x_data = torch.from_numpy(train_dataset_X).reshape(-1, 64*64*3).float() / 256
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
        self.x_data = torch.from_numpy(test_dataset_X).reshape(-1, 64*64*3).float() / 256
        self.y_data = torch.from_numpy(test_dataset_y).reshape(-1, 1).float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
        

class Net(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_features, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.fc4 = torch.nn.Linear(256, 128)
        self.fc5 = torch.nn.Linear(128, 64)
          self.fc6 = torch.nn.Linear(64, 1)

    def forward(self, X):
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))
        X = torch.relu(self.fc3(X))
        X = torch.relu(self.fc4(X))
        X = torch.relu(self.fc5(X))
        X = torch.sigmoid(self.fc6(X))

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

        
if name == '__main__':
    # Data Loader
    train_data = TrainDataset()
    train_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)

    test_data = TestDataset()
    test_loader = DataLoader(dataset=test_data, batch_size=10, shuffle=True)

    # Model Instance
    torch.manual_seed(0)
    net = Net(N_FEATURES)
    net.train(train_loader, EPOCHS, LEARNING_RATE, VERBOSE)

    train_accuracy = net.calculate_accuracy(train_loader)
    print(f"\nTrain accuracy after {EPOCHS} epochs : {round(train_accuracy, 2)}")

    test_accuracy = net.calculate_accuracy(test_loader)
    print(f"\nTest accuracy after {EPOCHS} epochs : {round(test_accuracy, 2)}\n")





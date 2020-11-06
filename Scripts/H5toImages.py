import os
import h5py
from PIL import Image
import numpy as np


train_dataset = h5py.File('../Datasets/TrainCatsVsNoncats.h5', "r")
train_dataset_X = np.array(train_dataset["train_set_x"][:])
train_dataset_y = np.array(train_dataset["train_set_y"][:])


# Make Directories
os.mkdir(os.path.join('../Datasets/', 'TrainImages'))
os.mkdir(os.path.join('../Datasets/', 'TestImages'))
os.mkdir(os.path.join('../Datasets/TrainImages/', 'Cats'))
os.mkdir(os.path.join('../Datasets/TrainImages/', 'Noncats'))
os.mkdir(os.path.join('../Datasets/TestImages/', 'Cats'))
os.mkdir(os.path.join('../Datasets/TestImages/', 'Noncats'))


cat_count = 1
noncat_count = 1

for image, label in zip(train_dataset_X, train_dataset_y):
    if label == 1:
        img = Image.fromarray(image)
        img.save('../Datasets/TrainImages/Cats/' + str(cat_count) + '.png')
        cat_count += 1
    else:
        img = Image.fromarray(image)
        img.save('../Datasets/TrainImages/Noncats/' + str(noncat_count) + '.png')
        noncat_count += 1


test_dataset = h5py.File('../Datasets/TestCatsVsNoncats.h5', "r")
test_dataset_X = np.array(test_dataset["test_set_x"][:])
test_dataset_y = np.array(test_dataset["test_set_y"][:])


cat_count = 1
noncat_count = 1

for image, label in zip(test_dataset_X, test_dataset_y):
    if label == 1:
        img = Image.fromarray(image)
        img.save('../Datasets/TestImages/Cats/' + str(cat_count) + '.png')
        cat_count += 1
    else:
        img = Image.fromarray(image)
        img.save('../Datasets/TestImages/Noncats/' + str(noncat_count) + '.png')
        noncat_count += 1


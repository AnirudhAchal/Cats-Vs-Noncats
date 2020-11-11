import os
import glob
from numpy import expand_dims
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot



# Data Generator
datagen = ImageDataGenerator(
    height_shift_range=0.2,
    width_shift_range=0.2,
    horizontal_flip=True,
    rotation_range=30
)


def train_DA():
    # Constant Variables
    TRAIN_CAT_DIR = "../Datasets/TrainImages/Cats/"
    TRAIN_CAT_COUNT = len(glob.glob(os.path.join(TRAIN_CAT_DIR, '*')))

    TRAIN_NONCAT_DIR = "../Datasets/TrainImages/Noncats/"
    TRAIN_NONCAT_COUNT = len(glob.glob(os.path.join(TRAIN_NONCAT_DIR, '*')))

    # Data Augmentation on Train Cats
    for filename in os.listdir(TRAIN_CAT_DIR):
        img = load_img(os.path.join(TRAIN_CAT_DIR, filename))
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        iterator = datagen.flow(samples, batch_size=1)

        for i in range(30):
            batch = iterator.next()
            image = batch[0].astype('uint8')
            img = Image.fromarray(image)
            img.save(os.path.join(TRAIN_CAT_DIR, str(TRAIN_CAT_COUNT) + '.png'))
            TRAIN_CAT_COUNT += 1


    # Data Augmentation on Train Noncats
    for filename in os.listdir(TRAIN_NONCAT_DIR):
        img = load_img(os.path.join(TRAIN_NONCAT_DIR, filename))
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        iterator = datagen.flow(samples, batch_size=1)

        for i in range(30):
            batch = iterator.next()
            image = batch[0].astype('uint8')
            img = Image.fromarray(image)
            img.save(os.path.join(TRAIN_NONCAT_DIR, str(TRAIN_NONCAT_COUNT) + '.png'))
            TRAIN_NONCAT_COUNT += 1


def test_DA():
    # Constant Variables
    TEST_CAT_DIR = "../Datasets/TestImages/Cats/"
    TEST_CAT_COUNT = len(glob.glob(os.path.join(TEST_CAT_DIR, '*')))

    TEST_NONCAT_DIR = "../Datasets/TestImages/Noncats/"
    TEST_NONCAT_COUNT = len(glob.glob(os.path.join(TEST_NONCAT_DIR, '*')))

    # Data Augmentation on Test Cats
    for filename in os.listdir(TEST_CAT_DIR):
        img = load_img(os.path.join(TEST_CAT_DIR, filename))
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        iterator = datagen.flow(samples, batch_size=1)

        for i in range(30):
            batch = iterator.next()
            image = batch[0].astype('uint8')
            img = Image.fromarray(image)
            img.save(os.path.join(TEST_CAT_DIR, str(TEST_CAT_COUNT) + '.png'))
            TEST_CAT_COUNT += 1


    # Data Augmentation on Test Noncats
    for filename in os.listdir(TEST_NONCAT_DIR):
        img = load_img(os.path.join(TEST_NONCAT_DIR, filename))
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        iterator = datagen.flow(samples, batch_size=1)

        for i in range(30):
            batch = iterator.next()
            image = batch[0].astype('uint8')
            img = Image.fromarray(image)
            img.save(os.path.join(TEST_NONCAT_DIR, str(TEST_NONCAT_COUNT) + '.png'))
            TEST_NONCAT_COUNT += 1


train_DA()

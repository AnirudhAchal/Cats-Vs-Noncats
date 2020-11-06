import os
import glob
from numpy import expand_dims
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot


# Constant Variables
CAT_DIR = "../Datasets/TrainImages/Cats/"
CAT_COUNT = len(glob.glob(os.path.join(CAT_DIR, '*')))

NONCAT_DIR = "../Datasets/TrainImages/Noncats/"
NONCAT_COUNT = len(glob.glob(os.path.join(NONCAT_DIR, '*')))


# Data Generator
datagen = ImageDataGenerator(
    height_shift_range=0.2,
    width_shift_range=0.2,
    horizontal_flip=True,
    rotation_range=30
)


# Data Augmentation on Cats
for filename in os.listdir(CAT_DIR):
    img = load_img(os.path.join(CAT_DIR, filename))
    data = img_to_array(img)
    samples = expand_dims(data, 0)
    iterator = datagen.flow(samples, batch_size=1)

    for i in range(30):
        batch = iterator.next()
        image = batch[0].astype('uint8')
        img = Image.fromarray(image)
        img.save(os.path.join(CAT_DIR, str(CAT_COUNT) + '.png'))
        CAT_COUNT += 1


# Data Augmentation on Noncats
for filename in os.listdir(NONCAT_DIR):
    img = load_img(os.path.join(NONCAT_DIR, filename))
    data = img_to_array(img)
    samples = expand_dims(data, 0)
    iterator = datagen.flow(samples, batch_size=1)

    for i in range(30):
        batch = iterator.next()
        image = batch[0].astype('uint8')
        img = Image.fromarray(image)
        img.save(os.path.join(NONCAT_DIR, str(NONCAT_COUNT) + '.png'))
        NONCAT_COUNT += 1


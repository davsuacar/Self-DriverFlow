
# coding: utf-8

import scipy.misc
from sklearn.utils import shuffle
import numpy as np


INDEX_TRAIN_BATH = 0
INDEX_TEST_BATH = 0
DATASET_FOLDER_PATH = '../images/dataset.csv'
IMAGES_FOLDER_PATH = '../images/images/'

df_data = []
df_label = []

with open(DATASET_FOLDER_PATH, 'r') as file:
    i = 0
    for line in file:
        df_data.append(IMAGES_FOLDER_PATH + "image_" + str(i) + '.png')
        df_label.append(line)
        i += 1

# Inputs
LIMIT = 1000

df_data, df_label = shuffle(df_data, df_label, random_state=22)

X_train = df_data
X_test = X_train[LIMIT:]
X_train = X_train[:LIMIT]

# Outputs
Y_train = df_label
Y_test = Y_train[LIMIT:]
Y_train = Y_train[:LIMIT]

num_train_images = len(X_train)
num_test_images = len(X_test)


def get_train_batch(batch_size):
    global INDEX_TRAIN_BATH

    x_out = []
    y_out = []
    for i in range(0, batch_size):
        scipy.misc.imsave("images_resized/test" + str(i) + ".png", (
            scipy.misc.imresize(scipy.misc.imread(X_train[(INDEX_TRAIN_BATH + i) % num_train_images]),
                                [64, 224, 3]) / 255.0))
        x_out.append(np.array(scipy.misc.imresize(scipy.misc.imread(X_train[(INDEX_TRAIN_BATH + i) % num_train_images])[-150:], [64, 224, 3]) / 255.0)[:,:,:3])
        y_out.append([Y_train[(INDEX_TRAIN_BATH + i) % num_train_images]])
    INDEX_TRAIN_BATH += batch_size
    return x_out, y_out


def get_test_batch(batch_size):
    global INDEX_TEST_BATH
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(np.array(scipy.misc.imresize(scipy.misc.imread(X_test[(INDEX_TEST_BATH + i) % num_test_images])[-150:], [64, 224, 3]) / 255.0)[:,:,:3])
        y_out.append([Y_test[(INDEX_TEST_BATH + i) % num_test_images]])
    INDEX_TEST_BATH += batch_size
    return x_out, y_out
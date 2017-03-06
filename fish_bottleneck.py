import os, cv2, random
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Activation
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

K.set_image_dim_ordering('th')

TRAIN_DIR = 'train/'
TEST_DIR = 'test_stg1/'
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
ROWS = 90  # 720
COLS = 160  # 1280
CHANNELS = 3


# Loading and Processing data

def get_images(fish):
    """Load files from train folder"""
    fish_dir = TRAIN_DIR + '{}'.format(fish)
    images = [fish + '/' + im for im in os.listdir(fish_dir)]
    return images


def read_image(src):
    """Read and resize individual images"""
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    return im.T


files = []
y_all = []

for fish in FISH_CLASSES:
    fish_files = get_images(fish)
    files.extend(fish_files)

    y_fish = np.tile(fish, len(fish_files))
    y_all.extend(y_fish)
    print("{0} photos of {1}".format(len(fish_files), fish))

y_all = np.array(y_all)

X_all = np.ndarray((len(files), CHANNELS, ROWS, COLS), dtype=np.uint8)

print X_all.shape

for i, im in enumerate(files):
    X_all[i] = read_image(TRAIN_DIR + im)

    if i % 1000 == 0: print('Processed {} of {}'.format(i, len(files)))

print(X_all.shape)

# splitting the training data

# One Hot Encoding Labels
y_all = LabelEncoder().fit_transform(y_all)
y_all = np_utils.to_categorical(y_all)

X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all,
                                                      test_size=0.2, random_state=23,
                                                      stratify=y_all)

# path to the model weights file.
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1./255)

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(CHANNELS, ROWS, COLS)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
    # assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'

    # use if not available in local
    '''
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                            TF_WEIGHTS_PATH, cache_subdir='models')'''


    weights_path = "weights/vgg16_weights.h5" # use this if you have it in local
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        print k, len(model.layers)
        if k >= 24:
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]

        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_datagen.fit(X_train)

    bottleneck_features_train = model.predict_generator(train_datagen.flow(X_train, batch_size=32), X_train.shape[0])

    print bottleneck_features_train.shape
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_datagen.fit(X_valid)

    bottleneck_features_validation = model.predict_generator(test_datagen.flow(X_valid, batch_size=32),
                                                             X_valid.shape[0])
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)


    # test set
    test_files = [im for im in os.listdir(TEST_DIR)]
    test = np.ndarray((len(test_files), CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, im in enumerate(test_files):
        test[i] = read_image(TEST_DIR + im)

    test_datagen.fit(test)

    bottleneck_features_test = model.predict_generator(test_datagen.flow(test, batch_size=32),
                                                             test.shape[0])
    np.save(open('bottleneck_features_test.npy', 'w'), bottleneck_features_test)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    # train_labels = np.array([0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))
    train_labels = y_train

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    # validation_labels = np.array([0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))
    validation_labels = y_valid

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='sigmoid'))

    #model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print "inside top model"
    print train_data.shape
    print train_labels.shape

    checkpointer = ModelCheckpoint(filepath="weights_fish2.h5", monitor='val_loss',
                                   verbose=1, save_best_only=True)
    model.fit(train_data, train_labels,
              nb_epoch=1, batch_size=32,
              validation_data=(validation_data, validation_labels),
              callbacks=[TensorBoard(log_dir='fish2'), checkpointer])

    model.save_weights(top_model_weights_path)


    # Predicting the test set

    test_data = np.load(open('bottleneck_features_test.npy'))
    test_preds = model.predict(test_data, verbose=1)
    test_files = [im for im in os.listdir(TEST_DIR)]
    submission = pd.DataFrame(test_preds, columns=FISH_CLASSES)
    submission.insert(0, 'image', test_files)
    submission.to_csv('sub_2.csv', index=False)


save_bottlebeck_features()
train_top_model()

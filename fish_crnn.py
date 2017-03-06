import os, cv2, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns


from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution1D, MaxPooling1D, ZeroPadding1D, Dense, Activation, LSTM
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

K.set_image_dim_ordering('tf')

TRAIN_DIR = 'train/'
TEST_DIR = 'test_stg1/'
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
ROWS = 92  #720
COLS = 160 #1280
CHANNELS = 3
# parameters
batch_size = 64
nb_epoch = 50
nb_classes = 8
nb_lstm_outputs = 128

# Loading and Processing data

def get_images(fish):
    """Load files from train folder"""
    fish_dir = TRAIN_DIR+'{}'.format(fish)
    images = [fish+'/'+im for im in os.listdir(fish_dir)]
    return images

def read_image(src):
    """Read and resize individual images"""
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (COLS,ROWS), interpolation=cv2.INTER_CUBIC)
    return im


files = []
y_all = []

for fish in FISH_CLASSES:
    fish_files = get_images(fish)
    files.extend(fish_files)
    
    y_fish = np.tile(fish, len(fish_files))
    y_all.extend(y_fish)
    print("{0} photos of {1}".format(len(fish_files), fish))
    
y_all = np.array(y_all)

X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, im in enumerate(files): 
    X_all[i] = read_image(TRAIN_DIR+im)
    if i%1000 == 0: print('Processed {} of {}'.format(i, len(files)))

print(X_all.shape)

# splitting the training data

# One Hot Encoding Labels
y_all = LabelEncoder().fit_transform(y_all)
y_all = np_utils.to_categorical(y_all)

X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, 
                                                    test_size=0.2, random_state=23, 
                                                    stratify=y_all)

X_train = X_train.reshape(X_train.shape[0], -1, 1)
X_valid = X_valid.reshape(X_valid.shape[0], -1, 1)
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_train /= 255
X_valid /= 255
print X_train.shape
print X_valid.shape


# Define model

def center_normalize(x):
    return (x - K.mean(x)) / K.std(x)

    
model = Sequential()
model.add(Activation(activation=center_normalize, input_shape=X_train.shape[1:]))
model.add(Convolution1D(nb_filter=32,
                        filter_length=3,
                        border_mode='same',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=2))
model.add(Convolution1D(nb_filter=64,
                        filter_length=3,
                        border_mode='same',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=2))
model.add(LSTM(nb_lstm_outputs))
model.add(Dense(nb_classes, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

checkpointer = ModelCheckpoint(filepath="weights_fish_crnn.h5", monitor='val_loss',
                               verbose=1, save_best_only=True)
# Fit the model on the batches.

model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose = 1, 
    validation_data=(X_valid, y_valid), shuffle=True, callbacks=[TensorBoard(log_dir='fish_crnn'), checkpointer])

# Predicting the test set
test_files = [im for im in os.listdir(TEST_DIR)]
test = np.ndarray((len(test_files), ROWS, COLS, CHANNELS), dtype=np.uint8)


for i, im in enumerate(test_files): 
    test[i] = read_image(TEST_DIR+im)

test = test.reshape(test.shape[0], -1, 1)
print test.shape
    
test_preds = model.predict(test, verbose=1)

submission = pd.DataFrame(test_preds, columns=FISH_CLASSES)
submission.insert(0, 'image', test_files)
submission.to_csv('sub_rnn.csv', index=False)
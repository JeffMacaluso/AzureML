"""
This script contains the code to train and save the model
"""

from __future__ import print_function
import os
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from azureml.logging import get_azureml_logger

run_logger = get_azureml_logger()

# Create the outputs folder - save any outputs you want managed by AzureML here
os.makedirs('./outputs', exist_ok=True)

batch_size = 128
num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Shortening the number of training examples
training_size = 20000
x_train = x_train[:training_size]
y_train = y_train[:training_size]

# Network Parameters
epochs = 1
dropout = 0.65
learning_rate = 0.001


# Tracking metrics
run_logger.log("Training Iterations", training_size)
run_logger.log("Epochs", epochs)
run_logger.log("Learning Rate", learning_rate)
run_logger.log("Dropout", dropout)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Logging the architecture used in the model
architecture = '3x3 -> 3x3 -> MaxPooling -> 128'
run_logger.log("Architecture", architecture)


# First layer - Conv 3x3
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

# Second layer - Conv 3x3
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout/2))

# Third layer - Dense 128
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(dropout))

# Output layer
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])

print('Beginning training')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,  # Change to 1 to display status bars
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])

# Logging 
run_logger.log("Test Loss", score[0])
run_logger.log("Test Accuracy", score[1])

# Saving the model
model.save('./outputs/model.h5')

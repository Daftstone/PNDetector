import os
import sys

import numpy as np
import keras
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from tensorflow.keras.layers import MaxPooling2D, Conv2D
from tensorflow.python.platform import flags
import pickle

FLAGS = flags.FLAGS


class MNIST_model:
    def __init__(self, input_shape=(None, 28, 28, 1), nb_filters=64, nb_classes=10):
        self.input_shape = input_shape
        self.nb_filters = nb_filters
        self.nb_classes = nb_classes
        self.model = self.get_model()

    def get_model(self):
        """
        define tohinz model
        :return:
        """
        model = tf.keras.Sequential()

        model.add(Conv2D(self.nb_filters, (3, 3), input_shape=self.input_shape[1:]))
        model.add(Activation('relu'))
        model.add(Conv2D(self.nb_filters, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(self.nb_filters * 2, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(self.nb_filters * 2, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))
        return model

    def train(self, x_train, y_train, x_test, y_test, batch_size=128, nb_epochs=60, is_train=True, is_umatrix=False):
        """
        detect adversarial examples
        :param x_train: train data
        :param y_train: train labels
        :param x_test:  test data
        :param y_test: test labels
        :param batch_size: batch size during training
        :param nb_epochs: number of iterations of model
        :param is_train: train online or load weight from file
        :return
        """
        optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        generator = ImageDataGenerator(rotation_range=15,
                                       width_shift_range=5. / 32,
                                       height_shift_range=5. / 32,
                                       horizontal_flip=True)
        generator.fit(x_train, seed=0)
        # Load model
        if (FLAGS.detection_type == 'negative'):
            weights_file = "weights/mnist/%s_model_%s.h5" % (FLAGS.detection_type, FLAGS.label_type)
        else:
            weights_file = "weights/mnist/origin_model.h5"
        if os.path.exists(weights_file) and is_train == False:
            self.model.load_weights(weights_file)
            print("Model loaded.")
            print("#############")

        lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                                       cooldown=0, patience=5, min_lr=1e-5)
        model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=False,
                                           save_weights_only=True, verbose=1)

        callbacks = [lr_reducer, model_checkpoint]
        if (is_train == True):
            print("begin train.")
            his = self.model.fit_generator(generator.flow(x_train, y_train, batch_size=batch_size),

                                           steps_per_epoch=len(x_train) // batch_size, epochs=nb_epochs,
                                           callbacks=callbacks,
                                           validation_data=(x_test, y_test),
                                           validation_steps=x_test.shape[0] // batch_size, verbose=1)
            with open('mnist_%s_%s' % (FLAGS.detection_type, FLAGS.label_type), 'wb') as file_pi:
                pickle.dump(his.history, file_pi)

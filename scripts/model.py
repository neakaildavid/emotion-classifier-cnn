"""
model.py

Creates the convolutional neural network architechture for 
facial emotion classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_emotion_model(input_shape, num_classes):
    """
    Builds and returns the CNN model for emotion classification
    
    :param input_shape (tuple): Shape of input images (H, W, C)
    :param num_classes (int): Number of emotion categories.

    :return tf.keras.Model: the constructed CNN model
    """
    model = models.Sequential()

    #Input
    model.add(layers.Input(shape=input_shape))

    #First Block
    model.add(layers.Conv2D(32, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.1))

    model.add(layers.Conv2D(32, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.1))

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    #Second Block
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.1))

    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.1))

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    #Third Block
    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.1))

    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.1))

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))

    #Fourth Block
    model.add(layers.Conv2D(256, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.1))

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))

    #Classifier
    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(num_classes, activation="softmax"))

    return model
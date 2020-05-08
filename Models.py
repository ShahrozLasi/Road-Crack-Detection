from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation



def conv_edge(n_classes = 6, IMG_SIZE = 128):
    model = Sequential([
                        ## First Layer
                        Conv2D(16, kernel_size = (3,3), kernel_initializer=tf.keras.initializers.he_uniform(seed=0),
                               input_shape=(IMG_SIZE, IMG_SIZE,3)),
                        BatchNormalization(),
                        Activation(activation='relu'),
                        MaxPooling2D((2,2), strides=(2,2)),
                        ## Second Layer
                        Conv2D(32, kernel_size=(3, 3), kernel_initializer=tf.keras.initializers.he_uniform(seed=0)),
                        BatchNormalization(),
                        Activation(activation='relu'),
                        MaxPooling2D((2, 2), strides=(2, 2)),
                        ## Third Layer
                        Conv2D(64, kernel_size=(3, 3), kernel_initializer=tf.keras.initializers.he_uniform(seed=0)),
                        BatchNormalization(),
                        Activation(activation='relu'),
                        MaxPooling2D((2, 2), strides=(2, 2)),
                        ## Forth Layer
                        Conv2D(128, kernel_size=(3, 3), kernel_initializer=tf.keras.initializers.he_uniform(seed=0)),
                        BatchNormalization(),
                        Activation(activation='relu'),
                        MaxPooling2D((2, 2), strides=(2, 2)),
                        ## Global Pool
                        GlobalAveragePooling2D(),
                        ## Dense
                        Dense(50, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0)),
                        Activation(activation='relu'),
                        Dropout(0.2),
                        ## Out Layer
                        Dense(n_classes, activation='softmax', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))
    ])

    return model

def conv_road(n_classes = 6, IMG_SIZE = 128):
    model = Sequential([
                        ## First Layer
                        Conv2D(16, kernel_size = (4,4), kernel_initializer=tf.keras.initializers.he_uniform(seed=0),
                               input_shape=(IMG_SIZE, IMG_SIZE,3)),
                        BatchNormalization(),
                        Activation(activation='relu'),
                        MaxPooling2D((2,2), strides=(2,2)),
                        ## Second Layer
                        Conv2D(32, kernel_size=(5, 5), kernel_initializer=tf.keras.initializers.he_uniform(seed=0)),
                        BatchNormalization(),
                        Activation(activation='relu'),
                        MaxPooling2D((2, 2), strides=(2, 2)),
                        ## Third Layer
                        Conv2D(64, kernel_size=(3, 3), kernel_initializer=tf.keras.initializers.he_uniform(seed=0)),
                        BatchNormalization(),
                        Activation(activation='relu'),
                        MaxPooling2D((2, 2), strides=(2, 2)),
                        ## Forth Layer
                        Conv2D(128, kernel_size=(4, 4), kernel_initializer=tf.keras.initializers.he_uniform(seed=0)),
                        BatchNormalization(),
                        Activation(activation='relu'),
                        MaxPooling2D((2, 2), strides=(2, 2)),
                        ## Flatten
                        Flatten(),
                        ## Dense
                        Dense(200, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0)),
                        Activation(activation='relu'),
                        Dropout(0.2),
                        ## Out Layer
                        Dense(n_classes, activation='softmax', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))
    ])

    return model

def vgg(n_classes= 6, IMG_SIZE = 128):
    from tensorflow.keras.applications import VGG16

    vgg16 = VGG16(weights='imagenet', include_top= False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    for layer in vgg16.layers:
        layer.trainable = False
    model = Sequential([vgg16,
                        Flatten(input_shape=(4, 4, 512)),
                        Dense(200, activation='relu', input_dim=(4 * 4 * 512),
                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0)),
                        Dropout(0.4),
                        Dense(n_classes, activation='softmax',
                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))
                        ])

    return model

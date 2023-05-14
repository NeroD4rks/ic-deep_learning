from tensorflow import keras
from tensorflow.keras import layers


def lenet(input_shape, n_classes):
    model = keras.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), input_shape=input_shape, activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(n_classes, activation='softmax'))

    return model
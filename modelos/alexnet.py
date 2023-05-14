from tensorflow import keras
from tensorflow.keras import layers


def alexnet(input_shape, n_classes):
    model = keras.Sequential()

    if input_shape[0] > 120:
        # Primeira camada de convolução

        model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

        # Segunda camada de convolução
        model.add(layers.Conv2D(256, (5, 5), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

        # Terceira camada de convolução
        model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))

        # Quarta camada de convolução
        model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))

        # Quinta camada de convolução
        model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

        # Camadas densas
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(n_classes, activation='softmax'))

    else:
        # Primeira camada de convolução
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

        # Segunda camada de convolução
        model.add(layers.Conv2D(192, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

        # Terceira camada de convolução
        model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))

        # Quarta camada de convolução
        model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))

        # Quinta camada de convolução
        model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Camadas densas
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(n_classes, activation='softmax'))

    return model

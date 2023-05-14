from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, ReLU, MaxPool2D, Add, \
    GlobalAvgPool2D, concatenate


def vgg(input_shape, num_classes):
    # reg=l2(1e-4)   # L2 or "ridge" regularisation
    reg2 = None
    num_filters2 = 32
    ac2 = 'relu'
    drop_dense2 = 0.5
    drop_conv2 = 0

    model2 = Sequential()

    model2.add(
        Conv2D(num_filters2, (3, 3), activation=ac2, kernel_regularizer=reg2, input_shape=input_shape, padding='same'))
    model2.add(BatchNormalization(axis=-1))
    model2.add(Conv2D(num_filters2, (3, 3), activation=ac2, kernel_regularizer=reg2, padding='same'))
    model2.add(BatchNormalization(axis=-1))
    model2.add(MaxPooling2D(pool_size=(2, 2)))  # reduces to 16x16x3xnum_filters
    model2.add(Dropout(drop_conv2))

    model2.add(Conv2D(2 * num_filters2, (3, 3), activation=ac2, kernel_regularizer=reg2, padding='same'))
    model2.add(BatchNormalization(axis=-1))
    model2.add(Conv2D(2 * num_filters2, (3, 3), activation=ac2, kernel_regularizer=reg2, padding='same'))
    model2.add(BatchNormalization(axis=-1))
    model2.add(MaxPooling2D(pool_size=(2, 2)))  # reduces to 8x8x3x(2*num_filters)
    model2.add(Dropout(drop_conv2))

    model2.add(Conv2D(4 * num_filters2, (3, 3), activation=ac2, kernel_regularizer=reg2, padding='same'))
    model2.add(BatchNormalization(axis=-1))
    model2.add(Conv2D(4 * num_filters2, (3, 3), activation=ac2, kernel_regularizer=reg2, padding='same'))
    model2.add(BatchNormalization(axis=-1))
    model2.add(MaxPooling2D(pool_size=(2, 2)))  # reduces to 4x4x3x(4*num_filters)
    model2.add(Dropout(drop_conv2))

    model2.add(Flatten())
    model2.add(Dense(512, activation=ac2, kernel_regularizer=reg2))
    model2.add(BatchNormalization())
    model2.add(Dropout(drop_dense2))
    model2.add(Dense(num_classes, activation='softmax'))

    return model2

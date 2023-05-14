from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, ReLU, MaxPool2D, Add, \
    GlobalAvgPool2D, concatenate
import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Input, Concatenate, Lambda


def resnet50(input_shape, num_class):
    def identity_block(input_tensor, filters, strides=1):
        f1, f2, f3 = filters

        x = Conv2D(filters=f1, kernel_size=(1, 1), strides=strides)(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(filters=f2, kernel_size=(3, 3), strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(filters=f3, kernel_size=(1, 1), strides=strides)(x)
        x = BatchNormalization()(x)

        x = Add()([x, input_tensor])
        output_tensor = ReLU()(x)

        return output_tensor

    def projection_block(input_tensor, filters, strides=2):
        f1, f2, f3 = filters
        x = Conv2D(filters=f1, kernel_size=(1, 1), strides=strides)(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(filters=f2, kernel_size=(3, 3), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(filters=f3, kernel_size=(1, 1), strides=1)(x)
        x = BatchNormalization()(x)

        # 1x1 conv projection shortcut
        shortcut = Conv2D(filters=f3, kernel_size=(1, 1), strides=strides)(input_tensor)
        shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        output_tensor = ReLU()(x)

        return output_tensor

    input = Input(shape=input_shape)

    x = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    x = projection_block(x, (64, 64, 256))
    x = identity_block(x, (64, 64, 256))
    x = identity_block(x, (64, 64, 256))

    x = projection_block(x, (128, 128, 512))
    x = identity_block(x, (128, 128, 512))
    x = identity_block(x, (128, 128, 512))
    x = identity_block(x, (128, 128, 512))

    x = projection_block(x, (256, 256, 1024))
    x = identity_block(x, (256, 256, 1024))
    x = identity_block(x, (256, 256, 1024))
    x = identity_block(x, (256, 256, 1024))
    x = identity_block(x, (256, 256, 1024))
    x = identity_block(x, (256, 256, 1024))

    x = projection_block(x, (512, 512, 2048))
    x = identity_block(x, (512, 512, 2048))
    x = identity_block(x, (512, 512, 2048))

    x = GlobalAvgPool2D()(x)
    x = Dense(num_class, activation='softmax')(x)

    model = tf.keras.Model(input, x, name='ResNet-50')
    return model

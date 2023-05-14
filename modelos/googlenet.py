from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, ReLU, MaxPool2D, Add, \
    GlobalAvgPool2D, concatenate
import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Input, Concatenate, Lambda
from tensorflow import keras


def googlenet(input_shape, num_class):
    def inception_module(prev, c1, c2, c3, c4):
        """
        Inception é o principal bloco da arquitetura do GoogleNet. O modulo Inception é uma construção local que empilha uns sobre os outros.
        Normalmente, a iniciação é feita de 4 caminhos paralelos
        """
        # O primeiro é feito de uma camada convolucional com kernel/ filter de 1x1. 1x1 convolucional reduz o caminho do canal
        p1 = Conv2D(filters=c1, kernel_size=(1, 1), activation='relu')(prev)
        p1 = BatchNormalization()(p1)
        # O segundo caminho composto de 2 camadas convolucionais sendo 1X1 e 3x3 do tamanho do filter. Novamente, a conv é feita antes da 3x3, para reduzir a dimensão de entrada por ser computacionalmente cara.
        p2 = Conv2D(filters=c2[0], kernel_size=(1, 1), activation='relu')(prev)
        p2 = BatchNormalization()(p2)
        p2 = Conv2D(c2[1], kernel_size=(3, 3), activation='relu', padding='same')(p2)
        p2 = BatchNormalization()(p2)

        # O terceiro caminho tem 1x1 e 5x% de camadas conv.
        p3 = Conv2D(filters=c3[0], kernel_size=(1, 1), activation='relu')(prev)
        p3 = BatchNormalization()(p3)
        p3 = Conv2D(filters=c3[1], kernel_size=(5, 5), activation='relu', padding='same')(p3)

        # O quarto caminho tem uma camada de maxpooling com pool sie de 3x3 e 1x1 conv.
        p4 = MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(prev)
        p4 = Conv2D(filters=c4, kernel_size=(1, 1), activation='relu')(p4)
        p4 = BatchNormalization()(p4)

        # Concatena os quatro caminhos
        output = concatenate(inputs=[p1, p2, p3, p4])

        return output

    input = keras.layers.Input(shape=input_shape)

    # Maiores modificação foram feitas aqui

    # Strides 1 para não realizar a divisão, que era possível ser realizada antes, já que era muito grande
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=1, padding='same', activation='relu')(input)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(x)

    x = inception_module(x, 64, (96, 128), (16, 32), 32)
    x = inception_module(x, 128, (128, 192), (32, 96), 64)
    x = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)

    x = inception_module(x, 192, (96, 208), (16, 48), 64)
    x = inception_module(x, 160, (112, 224), (24, 64), 64)
    x = inception_module(x, 128, (128, 256), (24, 64), 64)
    x = inception_module(x, 112, (144, 288), (32, 64), 64)
    x = inception_module(x, 256, (160, 320), (32, 128), 128)
    x = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

    x = inception_module(x, 256, (160, 320), (32, 128), 128)
    x = inception_module(x, 384, (192, 384), (48, 128), 128)
    x = AveragePooling2D(pool_size=(1, 1))(x)
    x = BatchNormalization()(x)
    ### Dropout & classification head

    # Para ser condizente com o tamanho, é encontrado a média aqui, talvez o pool_size mude, futuramente alterando os valores aqui
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)

    output = Dense(units=num_class, activation='softmax')(x)

    model = tf.keras.Model(input, output)
    return model

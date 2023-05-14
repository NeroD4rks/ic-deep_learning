from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, ReLU, MaxPool2D, Add, \
    GlobalAvgPool2D, concatenate
import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Input, Concatenate, Lambda
from tensorflow import keras


def densenet(input_shape, n_class):
    def dense_block(input_tensor, k, block_reps):
        """
        tensor: insere camadas anteriores
        k: grau de crescimento
        block_reps: Número de vezes que o bloco é repetido
        retorna os tensores concatenados
        """
        for _ in range(block_reps):
            x = BatchNormalization()(input_tensor)
            x = keras.layers.ReLU()(x)
            x = Conv2D(filters=4 * k, kernel_size=1)(x)
            x = BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            x = Conv2D(filters=k, kernel_size=3, padding='same')(x)

            output_tensor = keras.layers.Concatenate()([input_tensor, x])

            return output_tensor

    def transition_layers(input_tensor, theta=0.5):
        """
        input_tensor: Tensor que origina dos blocos densos (dense block)
        theta: fator de compressão, a ser multiplicado pelos mapas de saídas do bloco denso anterior
        retorna a saída do tensor

        """

        filters = input_tensor.shape[-1] * theta

        x = BatchNormalization()(input_tensor)
        x = keras.layers.ReLU()(x)
        x = Conv2D(filters=filters, kernel_size=2)(x)
        output_tensor = AveragePooling2D(pool_size=2, strides=2)(x)

        return output_tensor

    k = 12  # growth rate

    input = keras.layers.Input(shape=input_shape)
    x = keras.layers.BatchNormalization()(input)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(filters=2 * k, kernel_size=7, strides=1)(x)
    x = keras.layers.MaxPooling2D(pool_size=3, strides=1)(x)

    x = dense_block(x, 32, 6)
    x = transition_layers(x)

    x = dense_block(x, 32, 12)
    x = transition_layers(x)

    x = dense_block(x, 32, 32)
    x = transition_layers(x)

    x = dense_block(x, 32, 32)

    x = keras.layers.GlobalAveragePooling2D()(x)

    output = Dense(n_class, activation='softmax')(x)

    model = tf.keras.Model(input, output)

    return model

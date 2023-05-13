from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, ReLU, MaxPool2D, Add, \
    GlobalAvgPool2D, concatenate
import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Input, Concatenate, Lambda
from tensorflow import keras
from tensorflow.keras import layers


def vgg16_transfer_learning(_, num_class):
    # Carregando o modelo pretreinado
    # pode definir o tamanho do dataset antes, porém da erro com tamanhos muito pequenos
    base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', pooling='avg')
    # Como o dataset que trabalhamos tem o tamanho muito diferente do que é o
    # padrão do imagenet (que seria 224X224) temos que definir novas camadas para adequar
    model = keras.Sequential()
    # repete a primeira camada
    model.add(keras.layers.UpSampling2D())

    # adiciona o modelo pretreinado a segunda camada
    model.add(base_model)

    # Trecho do tutorial enviado, que é o caso dos datasets usados

    #   Conjunto de dados grande e diferente do modelo pré-treinado
    #     Adiciona-se camadas densamente conectas depois das camadas convolucionais;
    #     Treina-se a rede neural usando os pesos da ImageNet como valores inciais.

    model.add(Flatten())
    model.add(Dense(512, activation=('relu')))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation=('relu')))
    model.add(Dropout(0.2))

    # Como temos dez classes, a saida tem que ser correspondente
    model.add(Dense(num_class, activation=('softmax')))

    return model


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


def resnet50_transfer_learning(input_shape, num_class):
    # Carregando o modelo pretreinado
    base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', pooling='avg')

    # Como o dataset que trabalhamos tem o tamanho muito diferente do que é o
    # padrão do imagenet (que seria 224X224) temos que definir novas camadas para adequar
    model = keras.Sequential()
    # repete a primeira camada
    model.add(keras.layers.UpSampling2D())

    # adiciona o modelo pretreinado a segunda camada
    model.add(base_model)

    model.add(Flatten())
    model.add(Dense(512, activation=('relu')))

    model.add(Dropout(0.2))

    model.add(Dense(256, activation=('relu')))

    model.add(Dropout(0.2))

    # Como temos dez classes, a saida tem que ser correspondente
    model.add(Dense(num_class, activation=('softmax')))

    return model


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

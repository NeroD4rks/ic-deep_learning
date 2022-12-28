

import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Input, Dense, AveragePooling2D, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization , Activation, MaxPool2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from keras import regularizers
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, cv2
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from keras.regularizers import l1

def get_model_densenet(input_shape, n_class):
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
        x = BatchNormalization()(x)
        x = Conv2D(filters=filters, kernel_size=2)(x)
        x = BatchNormalization()(x)
        output_tensor = AveragePooling2D(pool_size=2, strides=2)(x)

        return output_tensor

    k = 12  # growth rate

    input = keras.layers.Input(shape=input_shape)
    #x = RandomFlip('horizontal') (input)
    #x = RandomZoom(0.2) (x)
    x = keras.layers.ReLU()(input)
    x = BatchNormalization()(x)
    x = keras.layers.Conv2D(filters=2 * k, kernel_size=7, strides=1)(x)
    x = BatchNormalization()(x)
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

def get_model_googlenet(input_shape, num_class):
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
    x = BatchNormalization() (x)
    x = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
    x = BatchNormalization() (x)
    x = Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization() (x)
    x = MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(x)

    x = inception_module(x, 64, (96, 128), (16, 32), 32)
    x = inception_module(x, 128, (128, 192), (32, 96), 64)
    x = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = BatchNormalization() (x)

    x = inception_module(x, 192, (96, 208), (16, 48), 64)
    x = inception_module(x, 160, (112, 224), (24, 64), 64)
    x = inception_module(x, 128, (128, 256), (24, 64), 64)
    x = inception_module(x, 112, (144, 288), (32, 64), 64)
    x = inception_module(x, 256, (160, 320), (32, 128), 128)
    x = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)


    x = inception_module(x, 256, (160, 320), (32, 128), 128)
    x = inception_module(x, 384, (192, 384), (48, 128), 128)
    x = AveragePooling2D(pool_size=(1, 1))(x)
    x = BatchNormalization() (x)
    ### Dropout & classification head

    # Para ser condizente com o tamanho, é encontrado a média aqui, talvez o pool_size mude, futuramente alterando os valores aqui
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = BatchNormalization() (x)
    x = Dropout(0.4)(x)
    
    x = Flatten()(x)
    
    output = Dense(units=num_class, activation='softmax')(x)

    model = tf.keras.Model(input, output)
    return model

def get_model_vgg16_transfer_learning(_, num_class):
    # Carregando o modelo pretreinado
    # pode definir o tamanho do dataset antes, porém da erro com tamanhos muito pequenos
    base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', pooling='avg')
    # Como o dataset que trabalhamos tem o tamanho muito diferente do que é o
    # padrão do imagenet (que seria 224X224) temos que definir novas camadas para adequar
    model = keras.Sequential()
    # repete a primeira camada
    model.add(BatchNormalization())
    model.add(keras.layers.UpSampling2D())

    # adiciona o modelo pretreinado a segunda camada
    model.add(base_model)

    # Trecho do tutorial enviado, que é o caso dos datasets usados

    #   Conjunto de dados grande e diferente do modelo pré-treinado
    #     Adiciona-se camadas densamente conectas depois das camadas convolucionais;
    #     Treina-se a rede neural usando os pesos da ImageNet como valores inciais.
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512, activation=('relu')))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(256, activation=('relu')))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    # Como temos dez classes, a saida tem que ser correspondente
    model.add(Dense(num_class, activation=('softmax')))

    return model


def get_model_resnet50_transfer_learning(_, num_class):
    # Carregando o modelo pretreinado
    base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', pooling='avg')

    # Como o dataset que trabalhamos tem o tamanho muito diferente do que é o
    # padrão do imagenet (que seria 224X224) temos que definir novas camadas para adequar
    model = keras.Sequential()
    # repete a primeira camada
    model.add(keras.layers.UpSampling2D())
    model.add(BatchNormalization())
    # adiciona o modelo pretreinado a segunda camada
    model.add(base_model)
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(512, activation=('relu')))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(256, activation=('relu')))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # Como temos dez classes, a saida tem que ser correspondente
    model.add(Dense(num_class, activation=('softmax')))

    return model

def get_model_alexnet(input_shape, num_class):
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(192, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(num_class, activation='softmax'))

    return model

def get_model_vgg(input_shape, num_class, weight_decay=0.0005):

    model = Sequential()
    

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dropout(0.256))

    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Dropout(0.15))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))

    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))

    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))


    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))

    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))


    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_class, activation='softmax'))

    return model

from keras.layers import Dense, Dropout, Flatten
import tensorflow as tf
from tensorflow import keras


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

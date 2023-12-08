from keras.layers import Dense, Dropout, Flatten
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def resnet50_transfer_learning(input_shape, num_class):
    # Carregando o modelo pretreinado
    base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', pooling='avg')

    # Como o dataset que trabalhamos tem o tamanho muito diferente do que é o
    # padrão do imagenet (que seria 224X224) temos que definir novas camadas para adequar
    model = keras.Sequential()

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                                           input_shape=(input_shape[0],
                                                        input_shape[1],
                                                        3)),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
        ]
    )
    # Create a new model
    model = keras.Sequential()
    model.add(data_augmentation)

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

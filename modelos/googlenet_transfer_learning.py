from keras.layers import Dense, Dropout, Flatten
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras import layers


def googlenet_transfer_learning(input_shape, num_classes):
    # Load the pre-trained model
    conv_layers = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet',
                                                                 include_top=False,
                                                                 input_tensor=tf.keras.layers.Input(input_shape),
                                                                 classes=num_classes)
    conv_layers.trainable = False
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

    model = tf.keras.Sequential([
        # data_augmentation,
        conv_layers,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

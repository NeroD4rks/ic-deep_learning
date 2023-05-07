import tensorflow as tf
import os
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random
import logging


def log_debug(txt: str, path: str = "file_log.log", printar=True):
    logging.basicConfig(filename=str(Path(path)), level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%d/%m/%Y %I:%M:%S %p')

    if printar:
        print(txt)
    logging.info(txt)


def preprocess(dataset, shape):
    """
    Para realizar a normalização do dataset
    """
    return np.array([tf.image.resize(img, size=[shape[0], shape[1]], method=tf.image.ResizeMethod.BICUBIC) for img in
                     dataset]) / 255


def augment(data_x, data_y, shape):
    """
    Função para realizar data augmentation, utilizado principalmente para evitar overffiting
    """
    array_y = []
    array_x = []
    for idx, image in enumerate(data_x):
        class_target = data_y[idx]
        image = tf.image.resize_with_crop_or_pad(image, shape[0] + 10, shape[1] + 10)  # Adiciona 10 pixels
        image = tf.image.random_crop(image, size=shape)  # Corta de volta
        image = tf.image.random_flip_left_right(image)  # aleatoriamente reflete
        image = tf.image.random_flip_up_down(image)  # aleatoriamente deixa de cabeça para baixo
        image = tf.clip_by_value(image, 0, 1)

        array_x.append(image)
        array_y.append(class_target)

    return array_x, array_y


def get_test_train(path: Path, shape):
    """"
    Função para pegar o test e train do dataset, dependendo da pasta que for passada
    """
    path = Path(path)
    x = []
    y = []
    classes = [int(c) for c in os.listdir(path)]

    for classe in classes:
        files = os.listdir(Path(f"{path}/{classe}"))

        for file in files:
            image = cv2.imread(str(Path(f"{path}/{classe}/{file}")))  # lendo a imagem
            image = cv2.resize(image, dsize=(shape[0], shape[1]))  # redimensionando
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # padrão rgb
            image = image / 255  # já fazendo um tratamento prévio (ainda n sei se é bom)
            x.append(image)  # adicionando ao dataset
            y.append(classe)  # adicionando a classe

    return x, y


def get_n_class(path: Path):
    """
    Pegar número de classes
    """
    return max([int(c) for c in os.listdir(path)]) + 1


def augment_images_save(path: Path, number: int, number_files: int = 1):
    """
    Função para realizar data augmentation salvando em arquivos, utilizado principalmente para evitar overffiting

    """
    path = Path(path)
    classes = [int(c) for c in os.listdir(path)]
    for classe in classes:
        files = []
        for _ in range(number_files):
            gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15,
                                     zoom_range=0.1,
                                     horizontal_flip=True)  # definindo as ações que serão realizadas, exemplo, rotacionar
            chosen_image = random.choice(list(
                set(os.listdir(str(Path(f"{path}/{classe}/")))) - set(files)))  # Para não repetir o que já foi usado
            if not chosen_image:
                break
            files.append(chosen_image)
            image_path = str(Path(f"{path}/{classe}/{chosen_image}"))  # selecionando uma imagem aleatoria
            image = np.expand_dims(plt.imread(image_path), 0)
            aug_iter = gen.flow(image, save_to_dir=f"{path}/{classe}/", save_prefix='aug-image-',
                                save_format='png')  # definindo pasta e prefixo (para deletar depois)
            log_debug(f"Gerando imagens a partir do arquivo {chosen_image} da classe {classe}")
            aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(
                number)]  # realizando efetivamente o data augmentation, gerando conforme o número informado
            log_debug("Terminando de gerar imagens\n\n")


def del_augment_images(path: Path):
    """
    Deletar imagens geradas a partir da função augment_images_save
    """
    path = Path(path)
    classes = [int(c) for c in os.listdir(path)]
    for classe in classes:
        log_debug(f"Deletando arquivos da classe: {classe}")
        files = [file for file in os.listdir(Path(f"{path}/{classe}")) if
                 "aug-image-" in file]  # selecionando para deletar se tiver o prefixo
        for file in files:
            os.remove(Path(f"{path}/{classe}/{file}"))  # deletando efetivamente

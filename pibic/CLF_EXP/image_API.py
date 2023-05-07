import sys
import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os
from PIL import Image
from pyts.image import MarkovTransitionField
from pyts.image import GramianAngularField
from pyts.image import RecurrencePlot
import shutil


def znorm(x):
    x_znorm = (x - np.mean(x)) / np.std(x)
    return x_znorm


def train_test_csv_into_np_array(dirpath, files):
    """Returns two 2d np arrays with train and test_datasets datasets of time series

    Keyword arguments:
    dirpath -- String with path to dir where the TRAIN and TEST .tsv are
    files -- List of strings with the contents of root
    """
    train = ""
    test = ""
    for file in files:
        if "TRAIN" in file:
            train = pd.read_csv(os.path.join(dirpath, file), header=None, delimiter="\t")
        elif "TEST" in file:
            test = pd.read_csv(os.path.join(dirpath, file), header=None, delimiter="\t")
    return train.to_numpy(), test.to_numpy()


def transform_dataset(series, colormap, path, img_number, type):
    """Converts a single 1d time-series array into an image

    Keyword arguments:
    series -- np array of a time series
    colormap -- string of a colormap
    path -- String with the incomplete path of the image output
    img_number -- Numbering of the image to be generated
    type --String with the image-generation algorithm name
    """
    series = znorm(series)  # Normalizes the time series
    cm = plt.get_cmap(colormap)
    # Selection of image-generation algorithm
    if type == "CWT":
        coeffs, freqs = pywt.cwt(series, scales=np.arange(1, len(series) + 1), wavelet='morl')
        im_final = Image.fromarray((cm(coeffs)[:, :, :3] * 255).astype(np.uint8))
    elif type == "MTF":
        series = series.reshape(1, len(series))
        mtf = MarkovTransitionField(n_bins=8)
        X_mtf = mtf.fit_transform(series)
        im_final = Image.fromarray((cm(X_mtf[0])[:, :, :3] * 255).astype(np.uint8))
    elif type == "GAF":
        series = series.reshape(1, len(series))
        gaf = GramianAngularField(method='summation')
        X_gaf = gaf.fit_transform(series)
        im_final = Image.fromarray((cm(X_gaf[0])[:, :, :3] * 255).astype(np.uint8))
    elif type == "GAF-DIFF":
        series = series.reshape(1, len(series))
        gaf = GramianAngularField(method='difference')
        X_gaf = gaf.fit_transform(series)
        im_final = Image.fromarray((cm(X_gaf[0])[:, :, :3] * 255).astype(np.uint8))
    elif type == "RP":
        series = series.reshape(1, len(series))
        rp = RecurrencePlot(threshold='distance')
        X_rp = rp.fit_transform(series)
        im_final = Image.fromarray((cm(X_rp[0])[:, :, :3] * 255).astype(np.uint8))
    elif type == "MTF-GAF-RP":
        series = series.reshape(1, len(series))
        mtf = MarkovTransitionField(n_bins=8)
        X_mtf = mtf.fit_transform(series)
        gaf = GramianAngularField(method='difference')
        X_gaf = gaf.fit_transform(series)
        rp = RecurrencePlot(threshold='distance')
        X_rp = rp.fit_transform(series)
        X_mgr = X_mtf[0] + X_gaf[0] + X_rp[0]
        im_final = Image.fromarray((cm(X_mgr)[:, :, :3] * 255).astype(np.uint8))
    elif type == "MTF-GAF":
        series = series.reshape(1, len(series))
        mtf = MarkovTransitionField(n_bins=8)
        X_mtf = mtf.fit_transform(series)
        gaf = GramianAngularField(method='difference')
        X_gaf = gaf.fit_transform(series)
        X_mg = X_mtf[0] - X_gaf[0]
        im_final = Image.fromarray((cm(X_mg)[:, :, :3] * 255).astype(np.uint8))

    file_name = type + "_" + colormap + "_" + str(img_number) + ".png"  # Builds the file name
    # Checks whether the output path already exists, and creates it if that's not the case
    if not os.path.isdir(path):
        os.makedirs(path)
    final_path = os.path.join(path, file_name)
    # print(final_path)
    im_final.save(final_path, "PNG", optimize=True, quality=95)
    im_final.close()


def transform_2darray_into_images(output_path, series_np, img_type, colormap):
    """Converts a whole 2d np array  into images

    Keyword arguments:
    output_path -- String with the root path where the images will be saved
    series_np -- Two-dimensional np array with several time series
    img_type -- String with the image-generation algorithm name
    colormap --string of a colormap
    """
    img_count = 1  # Couting to name each image
    for seq in series_np:
        serie_class = str(int(seq[0]))  # gets the class of the series
        path_class = os.path.join(output_path, serie_class)
        transform_dataset(seq, colormap, path_class, img_count, img_type)
        img_count += 1


def transform_train_and_test_into_images(img_type, colormap, input_path, output_path):
    """Transforms the train-test_datasets partitions of a single dataset into datasets of images

    Keyword arguments:
    input_path -- String with the path to the folder with the TRAIN.csv and TEST.csv
    output_path -- String with the path where the TRAIN and TEST image transformations are saved
    img_type -- String with the image-generation algorithm name
    colormap --string of a colormap
    """
    files = list(filter(lambda element: os.path.isfile(os.path.join(input_path, element)), os.listdir(input_path)))
    train_np, test_np = train_test_csv_into_np_array(input_path, files)
    train_output_path = os.path.join(output_path, "TRAIN")
    test_output_path = os.path.join(output_path, "TEST")
    transform_2darray_into_images(train_output_path, train_np, img_type, colormap)
    transform_2darray_into_images(test_output_path, test_np, img_type, colormap)


def transform_all_datasets_into_images(root_input, root_output):
    """Turns a group of time series datasets into a set of images

    Keyword arguments:
    root_input -- String with the root directory where the datasets are
    root_output -- String with the root directory where the image datasets will be created
    """
    cmap_list = ['binary', 'plasma', 'seismic', 'terrain', 'Paired']  # Paired tem que ser maiúsculo
    img_list = ["CWT", "MTF", "RP", "GAF", "GAF_DIFF", "MTF_GAF_RP"]
    dirs = list(filter(lambda element: os.path.isdir(os.path.join(root_input, element)), os.listdir(root_input)))
    for img in img_list:
        for colormap in cmap_list:
            start_output_path = os.path.join(root_output, img.upper() + "-" + colormap.upper())
            for dir in dirs:
                print(dir, img, colormap)
                input_path = os.path.join(root_input, dir)
                output_path = os.path.join(start_output_path, dir)
                transform_train_and_test_into_images(img, colormap, input_path, output_path)


def delete_image_dataset(img_dataset_path):
    """Deletes the folder of an image dataset

    Keyword arguments:
    img_dataset_path -- String with the path of the folder with the image dataset
    """
    if "CWT" or "MTF" or "RP" or "GAF" in img_dataset_path:
        shutil.rmtree(img_dataset_path, ignore_errors=True)





import numpy as np
import os
import ffmpeg
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import io


def VideoCompression(img1_path, img2_path, compressor):
    #Requer instalar FFMPEG no C\ e colocar no PATH
    """Encode two images with a CODEC compressor and return the compressed siz

    Keyword arguments:
    img1_path -- String with the path of the first image
    img2_path -- String with the path of the second image
    compressor -- String with the encoding CODEC
    """
    imgs = ffmpeg.concat(
        ffmpeg.input(img1_path, format='image2', vcodec='png'), #vcodec deve bater com o tipo de imagem
        ffmpeg.input(img2_path, format='image2', vcodec='png'),
    )
    stream = imgs.output('pipe:', format='rawvideo', vcodec=compressor)
    stream = stream.global_args( '-loglevel', 'quiet', ).run_async(pipe_stdout=True)
    compression = stream.stdout.read()
    return len(compression)

def CK_1(img1_path, img2_path, compressor):
    """Calculates CK-1 distance between two images using a specified encoding codec

    Keyword arguments:
    img1_path -- String with the path of the first image
    img2_path -- String with the path of the second image
    compressor -- String with the encoding CODEC
    """
    Cx_y = VideoCompression(img1_path, img2_path, compressor)
    Cy_x = VideoCompression(img2_path, img1_path, compressor)
    Cx_x = VideoCompression(img1_path, img1_path, compressor)
    Cy_y = VideoCompression(img2_path, img2_path, compressor)
    x = (Cx_y + Cy_x)/(Cx_x + Cy_y) -1
    return (Cx_y + Cy_x)/(Cx_x + Cy_y) -1

def loag_imag_into_matrix(imgs):
    matrix = []
    for image_path in imgs:
        img = io.imread(image_path, as_gray=False)
        matrix.append(img)
    return matrix


def scikitimage_metrics_1NN(X_train_matrix, y_train, test_instance):
    """1NN using as metrics Mean-Squared Error (MSE), normalized root mean square error (NRMSE),
    Structural Similarity Index (SSIM), Peak Signal to Noise Ratio (PSNR)

    Keyword arguments:
    X_train_matrix -- Array of strings with the paths of the images in the TRAIN partition
    y_train -- Array of strings with the classes of the images in the TRAIN partition
    test_instance -- String with the path of the to-be-classified image
    """
    #Index 0: MSE, Index 1: NRMSE, Index 2: SSIM, Index 3: PSNR
    predictions = {"mse":"", "nrmse":"", "ssim":"", "psnr":""}
    distances = {"mse":np.inf, "nrmse":np.inf, "ssim":np.inf *-1, "psnr":np.inf *-1}
    im1 = io.imread(test_instance, as_gray=False)
    for img_index in range(len(X_train_matrix)):
        im2 = X_train_matrix[img_index]
        current_distance_mse = mse(im1, im2)
        current_distance_nrmse = nrmse(im1, im2)
        #To compute colors with SSIM, either use multichannel=True or channel_axis = 2
        #current_distance_ssim = ssim(im1, im2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, channel_axis=2)
        current_distance_ssim = ssim(im1, im2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, multichannel=True)
        current_distance_psnr = psnr(im1, im2)
        if current_distance_mse<distances["mse"]: #smallest distance
            predictions["mse"] = y_train[img_index]
            distances["mse"] = current_distance_nrmse
        if current_distance_nrmse<distances["nrmse"]:  #smallest distance
            predictions["nrmse"] = y_train[img_index]
            distances["nrmse"] = current_distance_nrmse
        if current_distance_ssim>distances["ssim"]: #biggest distance
            predictions["ssim"] = y_train[img_index]
            distances["ssim"] = current_distance_ssim
        if current_distance_psnr>distances["psnr"]: #biggest distance
            predictions["psnr"] = y_train[img_index]
            distances["psnr"] = current_distance_psnr
    return predictions, distances

def run_dataset_scikit_metrics(X_test, y_test, X_train, y_train):
    """Runs a Dataset of images through a 1NN using MSE, NRMSE, SSIM and PSNR metrics

    Keyword arguments:
    X_test -- Array of strings with the paths of the images in the TEST partition
    y_test -- Array of strings with the classes of the images in the TEST partition
    X_train -- Array of strings with the paths of the images in the TRAIN partition
    y_train -- Array of strings with the classes of the images in the TRAIN partition
    """
    matches_per_metric = {"mse":0, "nrmse":0, "ssim":0, "psnr":0}
    all_predictions_per_metric = {"mse": [], "nrmse": [], "ssim": [], "psnr": []}
    best_distances_per_metric = {"mse": [], "nrmse": [], "ssim": [], "psnr": []}
    instance_number = len(X_test) #the quantity of test_datasets instances
    train_matrix = loag_imag_into_matrix(X_train)
    for test_instance_index in range(instance_number):
        test_instance = X_test[test_instance_index]
        predictions_per_metric, best_distance_per_metric = scikitimage_metrics_1NN(train_matrix, y_train, test_instance)
        for metric in predictions_per_metric: #checks each prediction against the correct classes for each metric
            all_predictions_per_metric[metric].append(predictions_per_metric[metric])
            best_distances_per_metric[metric].append(best_distance_per_metric[metric])
            if predictions_per_metric[metric] == y_test[test_instance_index]:
                matches_per_metric[metric] +=1
    for metric in matches_per_metric: #gets the accuracy for each metric
        matches_per_metric[metric] /= instance_number
    return matches_per_metric, all_predictions_per_metric,best_distances_per_metric

def CK1_1NN(X_train, y_train, test_instance, compressor):
    """1NN using CK_1 distance as a metric

    Keyword arguments:
    X_train -- Array of strings with the paths of the images in the TRAIN partition
    y_train -- Array of strings with the classes of the images in the TRAIN partition
    test_instance -- String with the path of the to-be-classified image
    compressor -- The enconding CODEC used
    """
    distance = np.inf
    predicted_class = ""
    for index in range(len(X_train)):
        current_distance = CK_1(test_instance, X_train[index], compressor)
        if current_distance<distance:
            predicted_class = y_train[index]
            distance = current_distance
    return predicted_class,distance

def mount_train_test(rootPath):
    """From a Folder with TEST and TRAIN folders (containing images divided into folders according to their class)
    builds 4 arrays of strings X_test, y_test, X_train, y_train

    Keyword arguments:
    rootPath -- String with the path containing a TEST and TRAIN folders
    """
    X_test = np.array([])
    X_train = np.array([])
    y_test = np.array([])
    y_train = np.array([])
    for root, dirs, files in os.walk(rootPath):
        if "TEST" in root and len(files)>0:
            c = root[len(root)-1] #Last char in the root is the class folder
            for file in files:
                X_test = np.append(X_test, np.array(os.path.join(root, file)))
                y_test = np.append(y_test, np.array(c))
        elif "TRAIN" in root and len(files)>0:
            c = root[len(root)-1] #Last char in the root is the class folder
            for file in files:
                X_train = np.append(X_train, np.array(os.path.join(root, file)))
                y_train = np.append(y_train, np.array(c))
    return X_test, y_test, X_train, y_train

def run_dataset_codec_acc(X_test, y_test, X_train, y_train, CODEC):
    """Runs a Dataset of images through a 1NN using CK_1 distance

    Keyword arguments:
    X_test -- Array of strings with the paths of the images in the TEST partition
    y_test -- Array of strings with the classes of the images in the TEST partition
    X_train -- Array of strings with the paths of the images in the TRAIN partition
    y_train -- Array of strings with the classes of the images in the TRAIN partition
    CODEC -- String with the path containing a TEST and TRAIN folders
    """
    predictions = []
    distances = []
    matches = 0
    instance_number = len(X_test)
    for test_instance_index in range(instance_number):
        #print(test_instance_index, CODEC)
        test_instance = X_test[test_instance_index]
        pred,dist = CK1_1NN(X_train, y_train, test_instance, CODEC)
        predictions.append(pred)
        distances.append(dist)
        if pred == y_test[test_instance_index]: matches +=1
    return matches/instance_number, predictions, distances
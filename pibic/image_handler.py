from functools import partial
import ffmpeg
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def VideoCompression(img1_path, img2_path, compressor='mpeg1video'):
    #Requer instalar FFMPEG no C\ e colocar no PATH
    """
    Encode two images with MPEG-1 compressor and return the compressed size.
    """
    imgs = ffmpeg.concat(
        ffmpeg.input(img1_path, format='image2', vcodec='png'), #vcodec deve bater com o tipo de imagem
        ffmpeg.input(img2_path, format='image2', vcodec='png'),
    )
    stream = imgs.output('pipe:', format='rawvideo', vcodec=compressor)
    stream = stream.global_args( '-loglevel', 'quiet', ).run_async(pipe_stdout=True)
    compression = stream.stdout.read()
    return len(compression)

def CK_1(img1_path, img2_path, compressor='mpeg1'):
    """
    Compute CK-1 distance with the corresponding compressor
    """
    #print(compressor, img1_path, img2_path)
    Cx_y = VideoCompression(img1_path, img2_path, compressor)
    Cy_x = VideoCompression(img2_path, img1_path, compressor)
    Cx_x = VideoCompression(img1_path, img1_path, compressor)
    Cy_y = VideoCompression(img2_path, img2_path, compressor)
    return (Cx_y + Cy_x)/(Cx_x + Cy_y) -1

def custom_dist_sklearn(a, b, compressor='mpeg1', images=[]):
    """
    Adapted custom distance function to work with sklearn KNN.
    Takes two indices and return the distance of the two
    images at those indices based on the given compressor.
    """

    comp_map = {'mpeg1video': partial(CK_1, compressor='mpeg1video'),
                'mpeg2video': partial(CK_1, compressor='mpeg2video'),
                'h264': partial(CK_1, compressor='h264'),
                'vp9': partial(CK_1, compressor='vp9'),
                'hevc': partial(CK_1, compressor='hevc'),
                }
    img1_path = images[int(a)]
    img2_path = images[int(b)]
    return comp_map[compressor](img1_path, img2_path)

def CK1_Sklearn(labels, Codec, lenTest, images):
  clf_ck1 = KNeighborsClassifier(n_neighbors=1, algorithm='brute',
                               metric=custom_dist_sklearn, metric_params={'compressor':Codec, 'images':images})
  indices = np.arange(len(labels))
  clf_ck1.fit(indices.reshape(-1, 1)[lenTest::], labels[lenTest::])
  return clf_ck1.score(indices[:lenTest:].reshape(-1, 1), labels[:lenTest:]), clf_ck1

def OneNN(X_train, y_train, image, compressor):
    distance = np.inf
    predicted_class = ""
    for index in range(len(X_train)):
        current_distance = CK_1(image, X_train[index], compressor)
        if current_distance<distance:
            predicted_class = y_train[index]
            distance = current_distance
    return predicted_class,distance
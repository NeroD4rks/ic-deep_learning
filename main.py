import os

from tensorflow import keras
from objects.datasets_Info import DatasetsInfo


def run(root_input, root_output):
    dt = DatasetsInfo()
    shape = (32, 32, 3)
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

                dt.iteration_dataset(output_path, representation, dir, shape)

                os.remove(output_path)
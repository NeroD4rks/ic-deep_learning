#!../env/bin/activate

import os
import shutil
import pandas as pd
from pathlib import Path

from objects.datasets_Info import DatasetsInfo
from pibic.CLF_EXP.image_API import transform_train_and_test_into_images


def run(root_input, root_output):
    dt = DatasetsInfo()
    if Path.exists(Path("results/alexnet.csv")):
        results = pd.read_csv(current_dir / f"results/alexnet.csv", delimiter=";", header=None)
    else:
        results = pd.DataFrame()

    shape = (32, 32, 3)
    cmap_list = ['binary', 'plasma', 'seismic', 'terrain', 'Paired']  # Paired tem que ser maiúsculo
    img_list = ["CWT", "MTF", "RP", "GAF", "GAF_DIFF", "MTF_GAF_RP"]
    dirs = list(filter(lambda element: os.path.isdir(os.path.join(root_input, element)), os.listdir(root_input)))
    for colormap in cmap_list:
        for img in img_list:
            start_output_path = os.path.join(root_output, img.upper() + "-" + colormap.upper())
            for dir in dirs:
                var = results[(results[0] == img) & (results[2] == colormap) & (results[1] == dir)]
                if not var.empty:
                    print(f"Ignorando {dir, img, colormap} por já ter sido executado")
                    continue

                input_path = os.path.join(root_input, dir)
                output_path = os.path.join(start_output_path, dir)
                print(f"Transformando em imagem o {dir, img, colormap}")
                transform_train_and_test_into_images(img, colormap, input_path, output_path)

                result_models = dt.iteration_dataset(output_path, img, dir, colormap, shape, 500)

                for model, result_model in result_models.items():
                    with open(current_dir / f'results/{model}.csv', 'a+') as file:
                        file.write(f'{img};{dir};{colormap};{result_model}\n')
                        file.close()

                shutil.rmtree(output_path)


if __name__ == '__main__':
    current_dir = Path.cwd()
    if not Path.exists(current_dir / "results/"):
        Path.mkdir(current_dir / "results/")

    run(current_dir / "UCRArchive_2018", current_dir / "datasets")

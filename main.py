#!../env/bin/activate
import sys
import os
import shutil
import pandas as pd
from pathlib import Path

from auxfunc import log_debug
from objects.datasets_Info import DatasetsInfo
from pibic.CLF_EXP.image_API import transform_train_and_test_into_images


def run(root_input, root_output):
    opcao = sys.argv[1] if len(sys.argv) > 0 else None
    dt = DatasetsInfo()
    if Path.exists(current_dir / "results/alexnet.csv"):
        results = pd.read_csv(current_dir / f"results/alexnet.csv", delimiter=";", header=None)
    else:
        results = pd.DataFrame()

    shape = (32, 32, 3)
    cmap_list = ['binary']  # Paired tem que ser maiúsculo

    if opcao and opcao == "exec1":
        img_list = ["CWT", "MTF", "RP"]
        log_debug(f"\nExecutando as representações  {img_list}")
    elif opcao and opcao == "exec2":
        img_list = ["GAF", "GAF_DIFF", "MTF_GAF_RP"]
        log_debug(f"\nExecutando as representações  {img_list}")
    else:
        log_debug(f"\nNenhuma opção informada, executando todas as representações")
        img_list = ["CWT", "MTF", "RP", "GAF"]

    dirs = ["features_originais", "dataset_22",  "dataset_resampled"]
    for dir in dirs:
        for colormap in cmap_list:
            for img in img_list:
                start_output_path = os.path.join(root_output, img.upper() + "-" + colormap.upper())
                if not results.empty:
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

    run(current_dir / "species_for_images", current_dir / "datasets")

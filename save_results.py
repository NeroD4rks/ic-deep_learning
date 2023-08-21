from pathlib import Path
import pandas as pd

files = [file for file in Path.iterdir(Path("results/"))]
dfs = []

for file in files:
    str_file = str(file)
    df = pd.read_csv(file, delimiter=";", names=["img", "colormap", "dataset"])
    df["arquitetura_cnn"] = str_file.replace("results/", "").split(".")[0]
    dfs.append(df)

df_final = pd.concat(dfs)
df_final.to_csv("save_results_instance2.csv")

# importacoes
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

root = Path('/content/drive/MyDrive/tcc/')
directory = root / "Rapid_age-grading_and_species_identification_of_natural_mosquitoes"
path_to_save = root / "age_for_images/dataset_resampled"
files_directory = os.listdir(directory)
path_to_save.mkdir()
cont = 0

dataset_resampled = pd.read_csv(directory / "dataset_resampled.csv", sep=";")

dataset_resampled['age'] = dataset_resampled['age'].map({'1-4': 0, '5-10': 1, '11-17': 2})
X = dataset_resampled.drop(['age', 'species'], axis=1)
y = dataset_resampled['age']

x_train, x_val, y_train, y_val = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)

# Inserir a nova coluna na posição 0
x_train.insert(0, 'age', y_train)

x_train.to_csv( path_to_save / 'TRAIN.csv', index=False, header=False)

# Inserir a nova coluna na posição 0
x_val.insert(0, 'age', y_val)

x_val.to_csv(path_to_save / 'TEST.csv', index=False, header=False)

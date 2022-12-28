from tensorflow import keras
from objects.datasets_Info import DatasetsInfo


dt = DatasetsInfo()

representation = "CWT-BINARY"

shape = (32,32,3)

dt.iteration_datasets(representation, shape)
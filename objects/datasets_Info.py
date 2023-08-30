from time import sleep

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tensorflow import keras
from keras.utils import np_utils
import keras as k
from auxfunc import *

from keras.preprocessing.image import ImageDataGenerator

from modelos.alexnet import alexnet
from modelos.googlenet import googlenet
from modelos.lenet import lenet
from modelos.resnet50_transfer_learning import resnet50_transfer_learning
from modelos.vgg_transfer_learning import vgg16_transfer_learning
from modelos.densenet import densenet


class DatasetsInfo:

    def __init__(self):
        self.df_results = None
        # Passando a referencia das funções para chamar posteriormente

        self.models = {
            "googlenet": googlenet,
            "resnet50_tf": resnet50_transfer_learning,
            "alexnet": alexnet,
        }

    def iteration_dataset(self, input_root: str, representation, type_image, colormap, shape, min_datasets):
        test_folder = Path(f"{input_root}/TEST")
        train_folder = Path(f"{input_root}/TRAIN")
        log_debug(f"\n\nAnalisando e processando ***{type_image}*** do {representation} de colormap {colormap}")

        n_class = get_n_class(train_folder)

        x_train_temp, y_train_temp = get_test_train(train_folder, shape)
        x_test, y_test = get_test_train(test_folder, shape)

        # Verificar o tamanho atual do conjunto de dados
        tamanho_atual = len(x_train_temp)

        if min_datasets > tamanho_atual:
            # Calcular quantas vezes o conjunto de dados precisa ser multiplicado
            multiplicador = int(np.ceil(min_datasets / tamanho_atual))

            # Multiplicar o conjunto de dados
            x_multiplicado = []
            y_multiplicado = []
            for _ in range(multiplicador):
                x_multiplicado.extend(x_train_temp)
                y_multiplicado.extend(y_train_temp)

            # Redimensionar o conjunto de dados para o tamanho desejado
            x_train = x_multiplicado[:min_datasets]
            y_train = y_multiplicado[:min_datasets]
        else:
            x_train, y_train = x_train_temp, y_train_temp

        y_train = k.utils.to_categorical(y_train, n_class)
        y_test = k.utils.to_categorical(y_test, n_class)

        log_debug("Iniciando preprocess, x_train")
        x_train = preprocess(x_train, shape)

        log_debug("Iniciando preprocess, x_test")
        x_test = preprocess(x_test, shape)

        results_model = self.execute_models(x_train, y_train, x_test, y_test, shape, n_class)

        return results_model

    def iteration_datasets(self, representation: str, shape) -> None:
        list_types = os.listdir(Path(f"datasets\{representation}"))
        results = []
        for type_image in list_types:
            row_to_add = {"representation": representation, "type_image": type_image}
            test_folder = f"datasets\{representation}\{type_image}\TEST"
            train_folder = f"datasets\{representation}\{type_image}\TRAIN"
            log_debug(f"\n\nAnalisando e processando ***{type_image}*** do {representation}")
            n_class = get_n_class(train_folder)
            x_train, y_train = get_test_train(train_folder, shape)
            x_test, y_test = get_test_train(test_folder, shape)

            y_train = np_utils.to_categorical(y_train)
            y_test = np_utils.to_categorical(y_test)

            log_debug("Iniciando preprocess, x_train")
            x_train = preprocess(x_train, shape)

            log_debug("Iniciando preprocess, x_test")
            x_test = preprocess(x_test, shape)

            results_model = self.execute_models(x_train, y_train, x_test, y_test, shape, n_class)
            for model, result_model in results_model.items():
                row_to_add[model] = result_model

            results.append(row_to_add)

        self.df_results = pd.DataFrame(result_model)

    def execute_models(self, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array, shape: tuple,
                       n_class: int) -> dict:

        """
        Função para efetuar o treinamento dos modelos e salvar os resultados
        """

        def decay(epoch):
            return 0.001 / (1 + 1 * epoch)

        results = {}

        datagen = ImageDataGenerator(
            rotation_range=10,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.15,
            zoom_range=0.1
        )

        datagen.fit(x_train, augment=True)

        callbacks = [keras.callbacks.LearningRateScheduler(decay, verbose=1),
                     #keras.callbacks.EarlyStopping(monitor='val_accuracy', baseline=1.0, patience=0)
                     ]

        for model_name in self.models:
            log_debug(f"Executando modelo {model_name}")

            model = self.models[model_name](shape, n_class)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            _ = model.fit(x=x_train, y=y_train,
                          validation_data=(x_test, y_test),
                          epochs=3,
                          callbacks=callbacks,
                          verbose=1)

            test_loss, test_acc = model.evaluate(x_test, y_test)

            y_pred_probabilities = model.predict(x_test)  # Probabilidades de classe para cada exemplo

            y_pred_class = np.argmax(y_pred_probabilities, axis=1)  # Classes previstas (índice do valor máximo)
            y_test_class = np.argmax(y_test, axis=1)
            # Calcular a matriz de confusão
            confusion = confusion_matrix(y_test_class, y_pred_class)

            # Calcular precisão, recall e F-measure
            precision = precision_score(y_test_class, y_pred_class, average='weighted')
            recall = recall_score(y_test_class, y_pred_class, average='weighted')
            f_measure = f1_score(y_test_class, y_pred_class, average='weighted')
            results[model_name] = test_acc

            msg = f"\nTerminou de executar o modelo: {model_name}\n"
            msg += f"Accuracy: {test_acc:.2f}\n"
            msg += f"Precision: {precision:.2f}\n"
            msg += f"Recall: {recall:.2f}\n"
            msg += f"F-measure: {f_measure:.2f}\n"
            msg += f"Confusion Matrix:\n {confusion}\n\n"

            log_debug(msg)

            sleep(5)

        return results

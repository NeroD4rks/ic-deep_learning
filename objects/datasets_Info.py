from tensorflow import keras
from keras.utils import np_utils
import keras as k
from auxfunc import *

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
            "densenet": densenet,
            "googlenet": googlenet,
            "vgg16_tf": vgg16_transfer_learning,
            "resnet50_tf": resnet50_transfer_learning,
            "alexnet": alexnet,
            "lenet": lenet
        }

    def iteration_dataset(self, input_root: str, representation, type_image, colormap, shape, multiple):
        test_folder = Path(f"{input_root}/TEST")
        train_folder = Path(f"{input_root}/TRAIN")
        log_debug(f"\n\nAnalisando e processando ***{type_image}*** do {representation}")

        n_class = get_n_class(train_folder)

        x_train, y_train = get_test_train(train_folder, shape)
        x_test, y_test = get_test_train(test_folder, shape)

        for i in range(multiple):
            x_train.extend(x_train)
            y_train.extend(y_train)

        x_test.extend(x_test)
        y_test.extend(y_test)

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

        for model_name in self.models:
            log_debug(f"Executando modelo {model_name}")
            callbacks = []
            callbacks += [keras.callbacks.LearningRateScheduler(decay, verbose=1)]

            model = self.models[model_name](shape, n_class)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            history = model.fit(x=x_train, y=y_train,
                                validation_data=(x_test, y_test),
                                epochs=10,
                                callbacks=callbacks,
                                verbose=1)

            test_loss, test_acc = model.evaluate(x_test, y_test)

            results[model_name] = test_acc

            log_debug(f"Terminou de executar o modelo: {model_name}\nresultado obtido: {test_acc}")

        return results

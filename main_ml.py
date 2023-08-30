from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing
import pandas as pd

import warnings

warnings.filterwarnings('ignore')


def log_debug(txt: str, path: str = "file_log.log", printar=True):
    logging.basicConfig(filename=str(Path(path)), level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%d/%m/%Y %I:%M:%S %p')

    if printar:
        print(txt)
    logging.info(txt)


parameters_KNN = {
    'n_neighbors': [1, 3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree'],
    'leaf_size': [5, 10, 20, 30],
    'p': [1, 2],
    'metric': ['euclidean', 'manhattan']
}

parameters_LR = {
    'penalty': [None, 'l1', 'l2', 'elasticnet'],
    'C': [0.1, 1.0, 5.0],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [100, 200, 300, 500],
    'class_weight': [None, 'balanced']
}

parameters_SVC = {
    'C': [0.1, 0.5, 1, 3, 5],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
    'class_weight': [None, 'balanced']
}
parameters_RF = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 3, 5, 7],
    'min_samples_split': [2, 5, 7, 10],
    'criterion': ['gini', 'entropy']
}
parameters_XGB = {
    'max_depth': [1, 3, 5, 7],
    'learning_rate': [0.1, 0.01],
    'n_estimators': [100, 200, 300],
    'gamma': [0, 0.1],
    'min_child_weight': [1, 3, 5]
}


def classificar(file, X_train, y_train, X_val, y_val) -> pd.DataFrame:
    clfKNN = KNeighborsClassifier()
    clfLR = LogisticRegression()
    clfSVC = SVC()
    clfRF = RandomForestClassifier()
    clfXGB = XGBClassifier()
    folds = 10
    # KNN
    gsKNN = GridSearchCV(clfKNN, parameters_KNN, scoring='accuracy', cv=folds, n_jobs=-1)
    gsKNN.fit(X_train, y_train)

    # LR
    gsLR = GridSearchCV(clfLR, parameters_LR, scoring='accuracy', cv=folds, n_jobs=-1)
    gsLR.fit(X_train, y_train)

    # SVM
    gsSVC = GridSearchCV(clfSVC, parameters_SVC, scoring='accuracy', cv=folds, n_jobs=-1)
    gsSVC.fit(X_train, y_train)

    # RF
    gsRF = GridSearchCV(clfRF, parameters_RF, scoring='accuracy', cv=folds, n_jobs=-1)
    gsRF.fit(X_train, y_train)

    # XGB
    gsXGB = GridSearchCV(clfXGB, parameters_XGB, scoring='accuracy', cv=folds, n_jobs=-1)
    gsXGB.fit(X_train, y_train)

    dfKNN = gsKNN.cv_results_
    dfLR = gsLR.cv_results_
    dfSVC = gsSVC.cv_results_
    dfRF = gsRF.cv_results_
    dfXGB = gsXGB.cv_results_

    log_debug("best params KNN: ")
    log_debug(gsKNN.best_params_)

    log_debug("best params LR: ")
    log_debug(gsLR.best_params_)

    log_debug("best params SVM: ")
    log_debug(gsSVC.best_params_)

    log_debug("best params RF: ")
    log_debug(gsRF.best_params_)

    log_debug("best params XGB: ")
    log_debug(gsXGB.best_params_)

    clfKNN = gsKNN.best_estimator_
    clfLR = gsLR.best_estimator_
    clfSVC = gsSVC.best_estimator_
    clfRF = gsRF.best_estimator_
    clfXGB = gsXGB.best_estimator_

    log_debug("\nKNN: \n")
    y_pred_knn = clfKNN.predict(X_val)
    acuracia_knn = accuracy_score(y_val, y_pred_knn)
    log_debug(f"Acuracia: {float(acuracia_knn):.5f} ")
    matriz_confusao_knn = confusion_matrix(y_val, y_pred_knn)
    log_debug("Matriz de Confusão - KNN:")
    log_debug(matriz_confusao_knn)

    log_debug("\nLR: \n")
    y_pred_lr = clfLR.predict(X_val)
    acuracia_lr = accuracy_score(y_val, y_pred_lr)
    log_debug(f"Acuracia: {float(acuracia_lr):.5f} ")
    matriz_confusao_lr = confusion_matrix(y_val, y_pred_lr)
    log_debug("Matriz de Confusão - LR:")
    log_debug(matriz_confusao_lr)

    log_debug("\nSVM: \n")
    y_pred_svc = clfSVC.predict(X_val)
    acuracia_svc = accuracy_score(y_val, y_pred_svc)
    log_debug(f"Acuracia: {float(acuracia_svc):.5f} ")
    matriz_confusao_svc = confusion_matrix(y_val, y_pred_svc)
    log_debug("Matriz de Confusão - SVM:")
    log_debug(matriz_confusao_svc)

    log_debug("\n RF: \n")
    y_pred_rf = clfRF.predict(X_val)
    acuracia_rf = accuracy_score(y_val, y_pred_rf)
    log_debug(f"Acuracia: {float(acuracia_rf):.5f} ")
    matriz_confusao_rf = confusion_matrix(y_val, y_pred_rf)
    log_debug("Matriz de Confusão - RF:")
    log_debug(matriz_confusao_rf)

    log_debug("\nXGB: \n")
    y_pred_xgb = clfXGB.predict(X_val)
    acuracia_xgb = accuracy_score(y_val, y_pred_xgb)
    log_debug(f"Acuracia: {float(acuracia_xgb):.5f} ")
    matriz_confusao_xgb = confusion_matrix(y_val, y_pred_xgb)
    log_debug("Matriz de Confusão - XGB:")
    log_debug(matriz_confusao_xgb)

    resultados = {
        'Classificador': ['KNN', 'Logistic Regression', 'SVC', 'Random Forest', 'XGBoost'],
        'Melhores Parâmetros': [gsKNN.best_params_, gsLR.best_params_, gsSVC.best_params_, gsRF.best_params_,
                                gsXGB.best_params_],
        'Melhor Acurácia': [acuracia_knn, acuracia_lr, acuracia_svc, acuracia_rf, acuracia_xgb]
    }

    df_resultados = pd.DataFrame(resultados)

    df_resultados.to_csv(file, sep=';', index=False)

    return df_resultados


def run_idades(_files):
    dfs = []
    for tuple_file in _files:
        file_rapid = tuple_file[0]
        file = str(file_rapid).split("/")[-1]
        file = path_results / f'new_age_resultados_classificadores_{file}.csv'
        file_ir = tuple_file[1]

        log_debug(f"\n\n\n\nLendo dataset {file_rapid}........")
        data_rapid = pd.read_csv(str(file_rapid) + ".csv", sep=';')
        data_rapid['age'] = data_rapid['age'].map({'1-4': 0, '5-10': 1, '11-17': 2})
        X = data_rapid.drop(['age', 'species'], axis=1)
        y = data_rapid['age']

        # log_debug(f"\n\n\n\nLendo dataset {file_ir}........")
        # data_ir = pd.read_csv(str(file_ir) + ".csv", sep=';')
        # data_ir['age'] = pd.cut(data_ir['age'], bins=[1, 5, 11, 18], labels=['1-4', '5-10', '11-17'], right=False)
        # data_ir['age'] = data_ir['age'].map({'1-4': 0, '5-10': 1, '11-17': 2})
        # X_val = data_ir.drop(['age', 'species'], axis=1)
        # y_val = data_ir['age']
        # separando uma parte para base de validação (30%)
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)

        df_gerado = classificar(file, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

        df_gerado["dataset"] = file_rapid
        dfs.append(df_gerado)

    merged_dataset = pd.concat(dfs, axis=1)
    merged_dataset.to_csv(path_results / f'new_age_resultados_classificadores.csv')


def run_species(_files):
    dfs = []
    log_debug(f"Classificando Species")
    for tuple_file in _files:
        file_rapid = tuple_file[0]
        file = str(file_rapid).split("/")[-1]
        file = path_results / f'new_species_resultados_classificadores_{file}.csv'
        file_ir = tuple_file[1]

        log_debug(f"\n\n\n\nLendo dataset {file_rapid}........")
        data = pd.read_csv(str(file_rapid) + ".csv", sep=';')
        X = data.drop(['age', 'species'], axis=1).copy()
        y = data['species'].map({'AR': 0, 'KS': 1}).copy()

        log_debug(f"\n\n\n\nLendo dataset {file_ir}........")
        # data = pd.read_csv(str(file_ir) + ".csv", sep=';')
        # X_val = data.drop(['age', 'species'], axis=1).copy()
        # y_val = data['species'].map({'AR': 0, 'KS': 1}).copy()
        # separando uma parte para base de validação (30%)
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)
        log_debug(f"\n\nX_val:\n{X_val.head(5)}")
        log_debug(f"y_val:\n{y_val.head(5)}")

        log_debug(f"X_test:\n{X_train.head(5)}")
        log_debug(f"y_test:\n{y_train.head(5)}\n\n")

        df_gerado = classificar(file, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

        df_gerado["dataset"] = file_rapid
        dfs.append(df_gerado)

    merged_dataset = pd.concat(dfs, axis=1)
    merged_dataset.to_csv(path_results / f'new_species_resultados_classificadores.csv')


root = Path.cwd()
path_rapid = Path(root / "Rapid/")
path_ir = Path(root / "IR/")
path_results = Path(root / "results/")

dataset = pd.DataFrame()
dataset_resampled = pd.DataFrame()

# dataset_22.csv  dataset_resampled.csv  features_originais.csv
files = [
    (Path(path_rapid / 'features_originais'), Path(path_ir / 'features_originais')),
    (Path(path_rapid / 'dataset_22'), Path(path_ir / 'dataset_22')),
    (Path(path_rapid / 'dataset_f22'), Path(path_ir / 'dataset_22')),
    (Path(path_rapid / 'dataset_resampled'), Path(path_ir / 'dataset_resampled'))
]

run_species(files[1:-1])
run_idades(files)
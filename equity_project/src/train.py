import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

from equity_project.src.utils import PurgedKFold, cpcv_paths

project_path = Path(__file__).parent.parent
HOLDING_PERIOD = 21 # сколько дней будем держать позицию
HP_CUTOFF = "2021-01-01" # дата, до которой будем подбирать гиперпараметры

# сетка гиперпараметров
PARAM_GRID = [
    {"iterations": 300, "depth": 4, "learning_rate": 0.05},
    {"iterations": 500, "depth": 6, "learning_rate": 0.03},
    {"iterations": 700, "depth": 6, "learning_rate": 0.02},
    {"iterations": 500, "depth": 8, "learning_rate": 0.03},
    {"iterations": 700, "depth": 8, "learning_rate": 0.02},
    {"iterations": 1000, "depth": 8, "learning_rate": 0.01},
]


def make_model(params):
    '''
    Создание модели CatBoost для оценки вероятности роста актива на опр. период
    '''
    return CatBoostClassifier(
        loss_function="Logloss", eval_metric="AUC",
        use_best_model=True, early_stopping_rounds=50,
        random_seed=42, verbose=False, **params,
    )


def train_model(X_tr, y_tr, params):
    '''
    Обучение модели при заданных гиперпараметрах на тренировочных данных
    '''
    cut = int(len(X_tr) * 0.8) # 80% - train, 20% - eval
    model = make_model(params) # создаем модель и учим ее
    model.fit(X=X_tr.iloc[:cut], y=y_tr.iloc[:cut], eval_set=(X_tr.iloc[cut:], y_tr.iloc[cut:]))
    return model


def select_hyperparams(X, y):
    '''
    Выбор лучших гиперпараметров на основе Purged K-Fold cross-validation на данных до HP_CUTOFF
    '''
    mask = X.index.get_level_values("Date") < pd.Timestamp(HP_CUTOFF) # берем даты до HP_CUTOFF
    X_hp, y_hp = X[mask], y[mask] 

    # получаем список фолдов для Purged K-Fold cross-validation
    splits = list(PurgedKFold(n_splits=5, horizon=HOLDING_PERIOD, embargo=HOLDING_PERIOD).split(X_hp))
    best_params = PARAM_GRID[0]
    best_auc = -np.inf # сравнивать модели будем по среднему auc на валидационных фолдах

    for params in PARAM_GRID: # проходим по всем комбинациям гиперпараметров
        aucs = []
        for tr, val in splits: # проходим по каждому фолду
            model = train_model(X_hp.iloc[tr], y_hp.iloc[tr], params)
            proba = model.predict_proba(X_hp.iloc[val])[:, 1]
            aucs.append(roc_auc_score(y_hp.iloc[val], proba))
        if (mean_auc := float(np.mean(aucs))) > best_auc:
            best_auc, best_params = mean_auc, params

    return best_params


def cpcv_evaluate(X, y, params, n_splits=6, n_test_splits=2):
    '''
    Оценка модели на основе Combinatorial Purged Cross-Validation на всех данных
    Возвращает словарь со средними метриками по всем путям
    Параметры:
        n_splits: количество фолдов для разбиения
        n_test_splits: количество тестовых фолдов в каждой комбинации
    '''
    paths = cpcv_paths(X, n_splits=n_splits, n_test_splits=n_test_splits, horizon=HOLDING_PERIOD, embargo=HOLDING_PERIOD)
    aucs, losses = [], []

    for tr_idx, test_idx in paths: # проходим по всем комбинациям тестовых фолдов
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        model = train_model(X_tr, y_tr, params)
        proba = model.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, proba))
        losses.append(log_loss(y_test, proba))

    return {
        "cpcv_mean_auc": float(np.mean(aucs)),
        "cpcv_mean_logloss": float(np.mean(losses)),
        "cpcv_n_paths": len(aucs),
    }


def train():
    '''
    Основная функция для обучения модели и ee сохранения
    '''
    os.makedirs(project_path / "models", exist_ok=True)

    # загружаем подготовленные данные
    X = pd.read_parquet(project_path / "data/processed/X_train.parquet")
    y = pd.read_parquet(project_path / "data/processed/y_train.parquet")["target"].astype(int)

    # удалаем пропуски
    valid = X.notna().all(axis=1)
    X, y = X[valid], y[valid]

    params = select_hyperparams(X, y) # выбираем лучшие гиперпараметры
    metrics = cpcv_evaluate(X, y, params) # оцениваем модель
    print(f"CPCV: {metrics}")

    # обучаем финальную модель на всех данных и сохраняем ее
    cut = int(len(X) * 0.9)
    model = train_model(X.iloc[:cut], y.iloc[:cut], params)
    joblib.dump(model, project_path / "models/model.joblib")


if __name__ == "__main__":
    train()

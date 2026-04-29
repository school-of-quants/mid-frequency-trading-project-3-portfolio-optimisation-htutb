import json
from itertools import combinations

import numpy as np
import pandas as pd
from yaml import safe_load


def load_config(config_path):
    '''
    Загрузка конфига из yaml файла
    '''
    with open(config_path) as f:
        return safe_load(f)


def save_dict(d, path):
    '''
    Сохранение словаря в json файл
    '''
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=4, default=str)


def mean_pairwise(m):
    '''
    Подсчет среднего значения верхнего треугольника матрицы
    '''
    upper = m.values[np.triu_indices(len(m), k=1)] # достаем верхний треугольник корреляционной матрицы без диагонали
    return pd.Series(upper).dropna().mean() # удаляем наны и считаем среднее


class PurgedKFold:
    '''
    Purged K-Fold cross-validator
    Возвращает индексы для обучения и валидации с учетом наличия embargo и purge во избежание утечки данных
    Параметры:
        n_splits: количество разбиений
        horizon: количество дней для purge 
        embargo: количество дней для embargo 
    '''
    def __init__(self, n_splits=5, horizon=21, embargo=5):
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo = embargo

    def split(self, X):
        dates = X.index.get_level_values(0) # достаем все даты из индекса 
        unique_dates = np.sort(dates.unique()) 
        n = len(unique_dates)
        fold_size = n // self.n_splits # получаем размер одного фолда

        for k in range(self.n_splits): # проходим по каждому фолду
            test_start = unique_dates[k * fold_size] # начало и конец тестового периода
            test_end = unique_dates[min((k + 1) * fold_size - 1, n - 1)]
            purge_start = unique_dates[max(np.searchsorted(unique_dates, test_start) - self.horizon, 0)]
            emb_end = unique_dates[min(np.searchsorted(unique_dates, test_end) + self.embargo, n - 1)]

            # определение масок для обучения и теста с учетом purge и embargo
            train_mask = (dates < purge_start) | (dates > emb_end)
            test_mask = (dates >= test_start) & (dates <= test_end)
            train_idx, test_idx = np.where(train_mask)[0], np.where(test_mask)[0]
            if len(train_idx) and len(test_idx):
                yield train_idx, test_idx


def cpcv_paths(X, n_splits=6, n_test_splits=2, horizon=21, embargo=21):
    '''
    Combinatorial Purged Cross-Validation.
    Возвращает список пар индексов для обучения и валидации для всех возможных комбинаций тестовых фолдов
    Параметры:
        n_splits: количество разбиений
        n_test_splits: количество тестовых фолдов в каждой комбинации
        horizon: количество дней для purge
        embargo: количество дней для embargo
    '''

    # создаем разбиение на фолды и сохраняем границы каждого фолда
    dates = X.index.get_level_values(0)
    unique_dates = np.sort(dates.unique())
    n = len(unique_dates)
    fold_size = n // n_splits
    edges = [(unique_dates[k * fold_size], unique_dates[min((k + 1) * fold_size - 1, n - 1)])
            for k in range(n_splits)]

    paths = []
    for test_groups in combinations(range(n_splits), n_test_splits): # проходим по всем возможным комбинациям тестовых фолдов
        test_mask = np.zeros(len(dates), dtype=bool)
        purge_mask = np.zeros(len(dates), dtype=bool)

        for g in test_groups: # для каждой комбинации
            ts, te = edges[g] # получаем границы тестового периода
            test_mask = test_mask | (dates >= ts) & (dates <= te) # обновляем маску тестового периода
            ps = unique_dates[max(np.searchsorted(unique_dates, ts) - horizon, 0)] # получаем границы purge периода
            ee = unique_dates[min(np.searchsorted(unique_dates, te) + embargo, n - 1)] # получаем границы embargo периода
            purge_mask = purge_mask | (dates >= ps) & (dates <= ee) & ~((dates >= ts) & (dates <= te)) # обновляем маску purge периода, исключая тестовый период

        # получаем индексы для обучения и теста, исключая purge период и сохраняем их в список
        train_idx = np.where(~test_mask & ~purge_mask)[0] 
        test_idx = np.where(test_mask)[0]
        if len(train_idx) and len(test_idx):
            paths.append((train_idx, test_idx))

    return paths


def compute_ipc(returns, window=60):
    '''
    Вычисление IPC (Intra-Portfolio Correlation) для заданного окна
    Параметры:
        window: кол-во дней для скользящего окна
    '''
    corr = returns.rolling(window, min_periods=window // 2).corr()
    return corr.groupby(level=0).apply(mean_pairwise) # группируем по дате и считаем среднее 

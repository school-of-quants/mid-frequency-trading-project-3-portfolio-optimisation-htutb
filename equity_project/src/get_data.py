import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from equity_project.src.utils import load_config

warnings.filterwarnings("ignore")

project_path = Path(__file__).parent.parent
HOLDING_PERIOD = 21 # сколько дней будем держать позицию (частота ребалансировки)
MIN_HISTORY = 252 # минимальное количество дней с данными для включения тикера в обучение
BETA_WINDOW = 60 # размер окна для вычисления беты
TRASH_TICKERS = {"DEC", "USBC", "CPWR", "TNB", "BMC", "SBNY"} # мусорные тикеры


# вспомогательные функции для обработки данных
def _load_pit(csv_path):
    '''
    Загрузка файла с историческими составами S&P 500, возврат датафреймов с тикерами на каждую дату изменения
    '''
    raw = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    raw["members"] = raw.iloc[:, 0].apply(
        lambda s: frozenset(t.strip().replace(".", "-") for t in str(s).split(",")) # чиним нейминг тикеров
    )
    return raw[["members"]]


def _members_on(pit, date):
    '''
    Возвращает тикеры S&P 500, актуальные на заданную дату
    '''
    idx = pit.index[pit.index <= date]
    return pit.at[idx[-1], "members"]


def _all_tickers(pit, start, end):
    '''
    Возвращает список всех тикеров, входивших в S&P 500 в определенный период
    '''
    mask = (pit.index >= start) & (pit.index <= end)
    universe = set()
    for members in pit.loc[mask, "members"]:
        universe |= set(members)
    return sorted(universe - TRASH_TICKERS)


def _date_slice(df, start, end):
    '''
    Фильтрация датафрейма по диапазону дат
    '''
    d = df.index.get_level_values("Date")
    return df[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]


def _ohlc_slice(raw, valid_tickers, start, end):
    '''
    Возвращает OHLCV данные для заданных тикеров и диапазона дат из
    '''
    mask = (raw.index >= pd.Timestamp(start)) & (raw.index <= pd.Timestamp(end))
    cols = pd.MultiIndex.from_product([["Close", "Open", "High", "Low", "Volume"], valid_tickers])
    return raw.loc[mask].reindex(columns=cols)


def generate_features(close, volume, spy_close, pit):
    '''
    Создание фичей для модели
    '''
    ret = close.pct_change() # дневные доходности активов
    spy_ret = spy_close.pct_change() # дневные доходности SPY

    spy_var = spy_ret.rolling(BETA_WINDOW).var() # скользящая дисперсия SPY
    spy_mat = pd.DataFrame(
        np.tile(spy_ret.values.reshape(-1, 1), (1, close.shape[1])),
        index=close.index, columns=close.columns,
    ) # матрица из доходностей SPY для вычисления ковариации с каждым активом
    beta = ret.rolling(BETA_WINDOW).cov(spy_mat).div(spy_var, axis=0) # скользящая бета для каждого актива

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    raw = {
        "mom_21": close.pct_change(21), # 21-дневный темп роста
        "mom_63": close.pct_change(63), # 63-дневный темп роста
        "mom_126": close.pct_change(126), # 126-дневный темп роста
        "rev_5": -close.pct_change(5), # реверсия за 5 дней
        "vol_20": ret.rolling(20).std(), # 20-дневная волатильность
        "vol_60": ret.rolling(60).std(), # 60-дневная волатильность
        "beta": beta, # бета относительно SPY
        "dist_ma50": (close - ma50) / close.replace(0, np.nan), # относительная разница между ценой и 50-дневной скользящей средней
        "dist_ma200": (close - ma200) / close.replace(0, np.nan), # относительная разница между ценой и 200-дневной скользящей средней
        "vol_trend": volume.pct_change(21), # 21-дневный темп роста объема торгов
    }

    features = {k: v.shift(1) for k, v in raw.items()} # сдвигаем фичи на 1 день вперед, чтобы избежать утечки данных

    # генерируем ранговые фичи для темпов роста
    for fname in ("mom_21", "mom_63", "mom_126"): # для каждой фичи темпа роста
        df = features[fname]
        ranked = df.copy() * np.nan
        for date in df.index: # для каждой даты
            pit_cols = [c for c in df.columns if c in _members_on(pit, date)] # берем существующие на эту даты тикеры
            ranked.loc[date, pit_cols] = df.loc[date, pit_cols].astype(float).rank(pct=True).values # расчитываем ранги (в процентах) на основе фичи 
        features[fname + "_rank"] = ranked # добавляем ранговую фичу

    # объединям все фичи в один датафрейм
    series = []
    for name, df in features.items():
        s = df.stack(future_stack=True)
        s.name = name
        series.append(s)

    X = pd.concat(series, axis=1)
    X.index.names = ["Date", "Ticker"]
    return X


def generate_labels(close, spy_close):
    '''
    Генерация таргета - бинарной метки (1 - актив растет сильнее SPY за следующий период, 0 - иначе)
    '''
    # расчитываем доходности за следующий период для каждого актива и SPY
    fwd_stock = close.pct_change(HOLDING_PERIOD).shift(-HOLDING_PERIOD) # темп роста актива за следующий период
    fwd_spy = spy_close.pct_change(HOLDING_PERIOD).shift(-HOLDING_PERIOD) # темп роста SPY за следующий период
    y = fwd_stock.gt(fwd_spy, axis=0).astype(int).stack(future_stack=True) # сравниваем темпы роста
    y.index.names = ["Date", "Ticker"]
    y.name = "target"
    return y


def get_data():
    '''
    Загрузка данных, генерация фичей/таргета, сохранение датасетов и данных для бэктеста
    '''
    # достаем из конфига даты 
    cfg = load_config(project_path.parent / "config.yaml")
    DOWNLOAD_START = cfg["download_start_date"]
    TRAIN_START = cfg["train_start_date"]
    TRAIN_END = cfg["train_end_date"]
    BT_START = cfg["backtest_start_date"]
    BT_END = cfg["backtest_end_date"]

    # скачиваем исторические данные для всех тикеров, которые входили в S&P 500, а также для SPY
    pit = _load_pit(project_path / "data/pony/S&P_500_Historical_Components.csv")
    tickers = _all_tickers(pit, TRAIN_START, BT_END)
    to_dl = sorted(set(tickers) | {"SPY"})

    raw = yf.download(to_dl, DOWNLOAD_START, BT_END, group_by="column", auto_adjust=True)
    raw.index = pd.to_datetime(raw.index)
    raw = raw.astype(float)
    if "Adj Close" in raw.columns.get_level_values(0): # удаляем столбцы со скорректированными ценами
        raw = raw.drop(columns="Adj Close", level=0)

    # достем цены закрытия, объем торгов (отдельно для SPY)
    close = raw["Close"].copy()
    volume = raw["Volume"].copy()
    spy_close = close.pop("SPY")
    volume.pop("SPY")

    for date in close.index: # для каждой даты
        members = _members_on(pit, date) # получаем актуальные тикеры
        non_members = [c for c in close.columns if c not in members]
        close.loc[date, non_members] = np.nan # для неактуальных тикеров ставим наны на цену и объем
        volume.loc[date, non_members] = np.nan

    train_mask = (close.index >= TRAIN_START) & (close.index <= TRAIN_END) # берем только даты для обучения
    valid_tickers = close.loc[train_mask].notna().sum()
    valid_tickers = valid_tickers[valid_tickers >= MIN_HISTORY].index.tolist() # оставляем только тикеры с достаточной историей данных
    close = close[valid_tickers]
    volume = volume[valid_tickers]

    X = generate_features(close, volume, spy_close, pit) # герерируем фичи
    X = X[X.index.get_level_values("Date") >= pd.Timestamp(TRAIN_START)] # оставляем только даты для обучения
    y = generate_labels(close, spy_close).reindex(X.index) # генерируем таргет

    # тренировочный датасет
    X_train = _date_slice(X, TRAIN_START, TRAIN_END) 
    y_train = _date_slice(y, TRAIN_START, TRAIN_END).dropna()
    X_train = X_train.reindex(y_train.index)

    # датасет для бэктеста
    X_backtest = _date_slice(X, BT_START, BT_END)

    for d in ("data/raw", "data/processed"):
        os.makedirs(project_path / d, exist_ok=True)

    # сохраняем данные
    _ohlc_slice(raw, valid_tickers, TRAIN_START, TRAIN_END).to_parquet(project_path / "data/raw/train_data.parquet")
    _ohlc_slice(raw, valid_tickers, BT_START, BT_END).to_parquet(project_path / "data/raw/backtest_data.parquet", engine="pyarrow")
    X_train.to_parquet(project_path / "data/processed/X_train.parquet")
    y_train.to_frame().to_parquet(project_path / "data/processed/y_train.parquet")
    X_backtest.to_parquet(project_path / "data/processed/X_backtest.parquet")


if __name__ == "__main__":
    get_data()

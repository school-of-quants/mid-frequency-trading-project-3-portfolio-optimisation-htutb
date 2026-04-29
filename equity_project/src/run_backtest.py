import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="vectorbt")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt

from equity_project.src.utils import load_config, save_dict, compute_ipc


project_path = Path(__file__).parent.parent
HOLDING_PERIOD = 21 # сколько дней будем держать позицию (частота ребалансировки)
TOP_N = 15 # макс. число активов, которые будем держать одновременно
ENTRY_THRESH = 0.30 # порог вероятности для входа в позицию
HOLD_THRESH = 0.20 # порог вероятности для удержания позиции
EMERGENCY_PROB = 0.10 # порог вероятности для ликвидации позиции 
STOP_LOSS = 0.10 # n-%ый стоп-лосс
MAX_WEIGHT = 0.10 # максимальный вес одного актива в портфеле
VOL_WINDOW = 20 # размер окна для расчета волатильности
IPC_WINDOW = 20 # размер окна для расчета IPC


def generate_weights(preds, backtest_data):
    '''
    Создание весов для портфеля на основе прогнозов модели и волатильности активов
    '''

    prob = preds.unstack(level=1)[1].fillna(0.0) # получаем вероятности роста актива
    close_all = backtest_data["Close"] # получаем цены закрытия 

    # рассчитываем волатильность для каждого актива
    vol = close_all.pct_change(fill_method=None).rolling(VOL_WINDOW).std().shift(1).reindex(prob.index)
    vol = vol.replace(0, np.nan).fillna(vol.median())

    # оставляем только те тикеры, для которых есть прогнозы и данные
    common = prob.columns.intersection(vol.columns)
    prob = prob[common]
    vol = vol[common]
    weights = pd.DataFrame(0.0, index=prob.index, columns=common) # создаем матрицу для весов

    rebal_dates = set(prob.index[::HOLDING_PERIOD]) # даты для ребалансировки портфеля
    held = {} # словарь для хранения текущих позиций

    for date in prob.index: # для каждой даты
        p_today = prob.loc[date] # получаем вероятность
        close_today = close_all[common].loc[date] # получаем цену закрытия

        for ticker in list(held): # для каждого тикера в текущем портфеле
            p = p_today.get(ticker, 0.0) # получаем вероятность
            if p < EMERGENCY_PROB or (date not in rebal_dates and p < HOLD_THRESH): # проверяем условия для ликвидации на основе вероятности
                del held[ticker]
                continue
            if (close_today[ticker] - held[ticker]) / held[ticker] < -STOP_LOSS: # проверяем условия для ликвидации на основе стоп-лосса
                del held[ticker]

        if date in rebal_dates: # на каждую дату ребалансировки формируем новый портфель
            candidates = set(p_today[p_today > ENTRY_THRESH].nlargest(TOP_N).index) # берем топ-N кандидатов для входа на основе вероятности
            survivors = {t for t in held if p_today.get(t, 0.0) >= HOLD_THRESH} # сохраняем выживших из текущего портфеля
            merged = sorted(survivors | candidates, key=lambda t: p_today.get(t, 0.0), reverse=True) # объединяем кандидатов и выживших
            held = {t: close_today[t] for t in merged[:TOP_N]} # обновляем текущий портфель (топ-N по вероятности)

        # формируем веса для текущего потрфеля
        tickers = list(held)
        raw_w = ((p_today[tickers].clip(lower=ENTRY_THRESH) - ENTRY_THRESH) # сдвигаем вероятности, чтобы порог входа - 0
                / vol.loc[date, tickers]).fillna(0.0) # вес - сдвинутая вероятность / волатильность
        if raw_w.sum() > 0:
            norm_w = (raw_w / raw_w.sum()).clip(upper=MAX_WEIGHT) # нормировка весов с учетом ограничения на максимальный вес
            weights.loc[date, tickers] = (norm_w / norm_w.sum()).values

    return weights


def run_backtest():
    '''
    Запуск бэктеста, сохранения результатов и метрик
    '''
    os.makedirs(project_path / "artifacts/plots", exist_ok=True)
    os.makedirs(project_path / "artifacts/metrics", exist_ok=True)

    cfg = load_config(project_path.parent / "config.yaml")

    # загружаем данные для бэктеста и обученную модель
    X_backtest = pd.read_parquet(project_path / "data/processed/X_backtest.parquet")
    backtest_data = pd.read_parquet(project_path / "data/raw/backtest_data.parquet", engine="pyarrow")

    model = joblib.load(project_path / "models/model.joblib")
    preds = pd.DataFrame(model.predict_proba(X_backtest), index=X_backtest.index)

    # формируем веса для портфеля и достаем цены
    close = backtest_data["Close"].dropna(axis=1, how="all")
    size = generate_weights(preds, backtest_data)

    cols = list(close.columns.intersection(size.columns))
    price = backtest_data.shift(-1)["Open"][cols] # исполняем сделки по цене открытия следующего дня
    close = close[cols]
    size = size[cols]

    pf = vbt.Portfolio.from_orders(
        close=close,
        price=price,
        size=size,
        size_type="targetpercent",
        group_by=True,
        cash_sharing=True,
        freq="1d",
        init_cash=cfg["init_cash"],
        fees=cfg["fees"],
    )

    # сохранение графика пнл и метри бэктеста
    pf.plot().write_image(str(project_path / "artifacts/plots/pnl.png"))
    save_dict(pf.stats().to_dict(), project_path / "artifacts/metrics/backtest_metrics.json")

    # расчет ipc для стратегии и бенчмарка, сохранение значений и графика динамики
    daily_ret = close.pct_change(fill_method=None).dropna(how="all")
    held_tickers = size.columns[size.max() > 0]
    ipc_eq = compute_ipc(daily_ret[held_tickers], window=IPC_WINDOW)
    ipc_bm = compute_ipc(daily_ret, window=IPC_WINDOW)

    ipc_df = pd.DataFrame({"strategy": ipc_eq, "benchmark": ipc_bm}).dropna(how="all")
    ipc_df.to_parquet(project_path / "artifacts/metrics/ipc.parquet")

    fig, ax = plt.subplots(figsize=(12, 4))
    ipc_df.plot(ax=ax, title=f"Rolling {IPC_WINDOW}-day IPC")
    ax.set_ylabel("Average pairwise correlation")
    fig.tight_layout()
    fig.savefig(str(project_path / "artifacts/plots/ipc.png"), dpi=150)
    plt.close(fig)

    save_dict(
        {"ipc_strategy_mean": float(ipc_eq.mean()), "ipc_benchmark_mean": float(ipc_bm.mean())},
        project_path / "artifacts/metrics/ipc_summary.json",
    )


if __name__ == "__main__":
    run_backtest()

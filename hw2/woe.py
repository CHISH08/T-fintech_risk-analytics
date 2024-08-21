import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from typing import Union, Tuple
from scipy.special import logit
import pandas as pd
def calc_buckets(x : Union[np.ndarray, pd.Series], n_buckets : int) -> np.ndarray:
    """Разбивает массив значений признака x на 
    n_buckets бакетов"""

    x = pd.Series(x).reset_index(drop=True)
    buckets = x.rank(method="dense", pct=True) * n_buckets
    buckets = np.ceil(buckets) - 1   # np.floor дает другой результат для 5.0, 6.0 и т.д.
    buckets = np.array(buckets, dtype=np.int16)

    return buckets
def woe_transform(badrate : float, offset : float) -> float:
    """Считаем WoE для бакета с данным badrate и выборки
    с данным offset."""
    epsilon = 0.001
    badrate = np.clip(badrate, epsilon, 1 - epsilon)
    woe = logit(badrate) - offset
    return woe
def woe_ci(target : np.ndarray, buckets : np.ndarray, offset : float) -> Tuple[pd.Series]:
    """Для каждого бакета вычисляем WoE и доверительный
    интервал для него."""
    # считаем бэдрейт и доверительный интервал для него (любым способом)
    z = 1.96

    _, indices = np.unique(buckets, return_inverse=True)
    bucket_n = np.bincount(indices)

    bucket_sum = np.bincount(indices, weights=target)
    badrate_i = bucket_sum / bucket_n

    delta_badrate = z * np.sqrt((badrate_i * (1 - badrate_i)) / bucket_n)
    badrate_lower = np.maximum(badrate_i - delta_badrate, 0)
    badrate_upper = np.minimum(badrate_i + delta_badrate, 1)

    woe = pd.Series(woe_transform(badrate_i, offset))
    woe_lower = pd.Series(woe_transform(badrate_lower, offset))
    woe_upper = pd.Series(woe_transform(badrate_upper, offset))


    return woe, woe_lower, woe_upper
def calc_line(values : np.ndarray, target : np.ndarray, mean_feature : np.ndarray, offset : float) -> np.ndarray:
    """Строим линейную интерполяцию для WoE."""

    # строим логистическую регрессию на одном признаке
    # и считаем ее предсказания в точках – mean_feature
    values = values.reshape((-1, 1))
    mean_feature = mean_feature.reshape((-1, 1))
    model = LogisticRegression()
    model.fit(values, target)
    proba = model.predict_proba(mean_feature)[:, 1]
    line = woe_transform(proba, offset)
    return line
def calc_buckets_info(values : np.ndarray, target : np.ndarray, buckets : np.ndarray) -> dict:
    """Для каждого бакета расчитывает
     - среднее значение признака
     - линейную интерполяцию в пространстве woe
     - значение woe и доверительный интервал для него"""

    offset = logit(target.sum() / len(target))

    buckets_info = {
        "mean_feature" : None,
        "line"         : None,
        "woe"          : None,
        "woe_lower"    : None,
        "woe_upper"    : None
    }

    buckets_info["woe"], buckets_info["woe_lower"], buckets_info["woe_upper"] = woe_ci(target, buckets, offset)

    buckets_info["mean_feature"] = np.bincount(buckets, weights=values) / np.bincount(buckets)

    buckets_info["line"] = calc_line(values, target, buckets_info["mean_feature"], offset)

    return buckets_info
def calc_plot_title(
        values : np.ndarray,
        target : np.ndarray,
        buckets : np.ndarray
    ) -> str:
    """Считает для признака roc auc, IV, R^2"""
    auc = roc_auc_score(target, values)

    info = calc_buckets_info(values, target, buckets)

    _, indices = np.unique(buckets, return_inverse=True)
    bucket_n = np.bincount(indices)
    bucket_sum = np.bincount(indices, weights=target)
    B_i = bucket_sum
    B = target.sum()
    G_i = bucket_n - B_i
    G = len(target) - B
    IV = np.sum((B_i / B - G_i / G) * info["woe"])

    # Взвешенный R^2
    # X - среднее в бакете, Y - woe в бакете, вес – число наблюдений в бакете
    weights = np.bincount(buckets)
    R_sqr = r2_score(info['woe'], info['line'], sample_weight=weights)

    plot_title = (        
        f"AUC = {auc:.3f} "
        f"IV = {IV:.3f} "
        f"R_sqr = {R_sqr:.3f} "
    )

    return plot_title
def make_figure(buckets_info : dict, plot_title : str) -> go.Figure:
    """Строит график линейности."""
    fig = go.Figure()

    # общие настройки
    title = dict(
        text=plot_title,
        y=0.95,
        x=0.5,
        font=dict(size=12),
        xanchor="center",
        yanchor="top"
    )
    margin = go.layout.Margin(
        l=50,
        r=50,
        b=50,
        t=60
    )

    fig.add_trace(
        go.Scatter(
            x=buckets_info["mean_feature"],
            y=buckets_info["line"],
            mode='lines',
            name="interpolation_line",
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=buckets_info["mean_feature"],
            y=buckets_info["woe"],
            line=dict(
                color='firebrick',
                width=1,
                dash='dot'
            ),
            error_y=dict(
                type='data',
                symmetric=False,
                array=buckets_info["woe_upper"],
                arrayminus=buckets_info["woe_lower"]
            ),
            name="WoE",
            showlegend=False)
    )

    fig.update_layout(
        width=1000,
        height=450,
        xaxis_title=dict(
            text='Feature value',
            font=dict(size=12)
        ),
        yaxis_title=dict(
            text="WoE",
            font=dict(size=12)
        ),
        title=title,
        margin=margin
    )

    return fig
def woe_line(
        values : np.ndarray,
        target : np.ndarray,
        n_buckets : int
    ) -> go.Figure:
    """График линейности переменной по WoE."""
    mask = ~np.isnan(values)
    values_without_nan = values[mask]
    target_without_nan = target[mask]
    buckets : np.ndarray = calc_buckets(values_without_nan, n_buckets)
    buckets_info : pd.DataFrame = calc_buckets_info(values_without_nan, target_without_nan, buckets)
    plot_title : str = calc_plot_title(values_without_nan, target_without_nan, buckets)
    fig = make_figure(buckets_info, plot_title)
    return fig
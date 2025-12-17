import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def _adf_pvalue(s: pd.Series) -> float:
    return float(adfuller(s.dropna(), autolag="AIC")[1])


def _find_d(s: pd.Series, max_d: int = 2, alpha: float = 0.05) -> int:
    x = s.copy()
    for d in range(max_d + 1):
        if _adf_pvalue(x) < alpha:
            return d
        x = x.diff()
    return max_d


def _valid_trends(trends: list[str], d: int) -> list[str]:
    # statsmodels rule: if d>0, 'c' (constant) is invalid
    if d > 0:
        vt = [t for t in trends if t != "c"]
        return vt if vt else ["n"]
    return trends if trends else ["c"]


def _rmse(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def _smape(a, b, eps=1e-12) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(100.0 * np.mean(2 * np.abs(a - b) / (np.abs(a) + np.abs(b) + eps)))


def _corr(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def select_arima_by_aic(train: pd.Series, p_max: int, q_max: int, trends: list[str]) -> dict:
    d = _find_d(train)
    vt = _valid_trends(trends, d)

    best = None
    best_aic = float("inf")

    for p in range(p_max + 1):
        for q in range(q_max + 1):
            for tr in vt:
                try:
                    model = ARIMA(
                        train,
                        order=(p, d, q),
                        trend=tr,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fit = model.fit(method_kwargs={"warn_convergence": False})
                    aic = float(fit.aic)
                    if aic < best_aic:
                        best_aic = aic
                        best = {"p": p, "d": d, "q": q, "trend": tr, "aic": aic}
                except Exception:
                    pass

    if best is None:
        raise RuntimeError("No valid ARIMA candidate found. Reduce p/q or check data.")
    return best


def rolling_one_step(train: pd.Series, test: pd.Series, order: tuple[int, int, int], trend: str) -> pd.Series:
    history = list(train.values)
    preds = []

    for true_value in test.values:
        model = ARIMA(
            history,
            order=order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(method_kwargs={"warn_convergence": False})
        yhat = float(fit.forecast(steps=1)[0])
        preds.append(yhat)
        history.append(float(true_value))

    return pd.Series(preds, index=test.index, name="ARIMA_pred")


def fit_predict_evaluate(
    y: pd.Series,
    test_horizon: int,
    p_max: int = 3,
    q_max: int = 3,
    trends: list[str] | None = None,
) -> tuple[pd.Series, dict]:
    if trends is None:
        trends = ["n", "c", "t"]

    train = y.iloc[:-test_horizon]
    test = y.iloc[-test_horizon:]

    best = select_arima_by_aic(train, p_max=p_max, q_max=q_max, trends=trends)
    order = (best["p"], best["d"], best["q"])

    pred_test = rolling_one_step(train, test, order=order, trend=best["trend"])

    metrics = {
        "RMSE": _rmse(test.values, pred_test.values),
        "MAE": _mae(test.values, pred_test.values),
        "sMAPE%": _smape(test.values, pred_test.values),
        "CORR": _corr(test.values, pred_test.values),
        "ORDER": order,
        "TREND": best["trend"],
        "AIC(train)": float(best["aic"]),
    }
    return pred_test, metrics

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


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


def fit_predict_evaluate(
    y: pd.Series,
    test_horizon: int,
    lags: int = 10,
    C: float = 1.0,
    epsilon: float = 0.1,
) -> tuple[pd.Series, dict]:
    if lags < 2:
        raise ValueError("lags must be >= 2")
    if test_horizon >= len(y) - lags:
        raise ValueError("test_horizon too large for given lags")

    train = y.iloc[:-test_horizon]
    test = y.iloc[-test_horizon:]

    # Build supervised set using TRAIN ONLY (no leakage)
    train_vals = train.values.astype(float)
    X_train, y_train = [], []
    for i in range(lags, len(train_vals)):
        X_train.append(train_vals[i - lags:i])
        y_train.append(train_vals[i])
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = SVR(kernel="linear", C=C, epsilon=epsilon)
    model.fit(X_train_s, y_train)

    # Rolling one-step on TEST (recursive with actual update)
    history = list(train.values.astype(float))
    preds = []

    for true_val in test.values.astype(float):
        x = np.asarray(history[-lags:], dtype=float).reshape(1, -1)
        x = scaler.transform(x)
        yhat = float(model.predict(x)[0])
        preds.append(yhat)
        history.append(float(true_val))

    pred_test = pd.Series(preds, index=test.index, name="SVR_pred")

    metrics = {
        "RMSE": _rmse(test.values, pred_test.values),
        "MAE": _mae(test.values, pred_test.values),
        "sMAPE%": _smape(test.values, pred_test.values),
        "CORR": _corr(test.values, pred_test.values),
        "LAGS": int(lags),
        "C": float(C),
        "epsilon": float(epsilon),
        "kernel": "linear",
    }
    return pred_test, metrics

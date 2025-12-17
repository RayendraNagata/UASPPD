import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


# =====================
# Metrics (samakan)
# =====================
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true, y_pred, eps=1e-12):
    return float(
        100 * np.mean(
            2 * np.abs(y_true - y_pred) /
            (np.abs(y_true) + np.abs(y_pred) + eps)
        )
    )


def corr(y_true, y_pred):
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


# =====================
# Windowing
# =====================
def make_windows(values, time_steps):
    X, y = [], []
    for i in range(time_steps, len(values)):
        X.append(values[i - time_steps:i])
        y.append(values[i])
    X = np.array(X).reshape(-1, time_steps, 1)
    y = np.array(y)
    return X, y


# =====================
# Main API (dipanggil app.py)
# =====================
def fit_predict_evaluate(
    y: pd.Series,
    test_horizon: int,
    time_steps: int = 60,
    epochs: int = 20,
):
    """
    Train LSTM on train split, evaluate rolling one-step on test split
    """

    values = y.values.astype(float).reshape(-1, 1)

    if len(values) < time_steps + test_horizon + 10:
        raise ValueError("Data terlalu pendek untuk LSTM")

    # ===== Split =====
    train_vals = values[:-test_horizon]
    test_vals = values[-test_horizon:]

    # ===== Scaling (train only) =====
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_vals)

    # ===== Windowing (TRAIN) =====
    X_train, y_train = make_windows(train_scaled, time_steps)

    # ===== Model =====
    model = Sequential([
        LSTM(32, input_shape=(time_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        verbose=0,
        callbacks=[es]
    )

    # ===== Rolling one-step on TEST =====
    history = train_vals.flatten().tolist()
    preds = []

    for true_val in test_vals.flatten():
        window = np.array(history[-time_steps:]).reshape(-1, 1)
        window_scaled = scaler.transform(window)
        x_input = window_scaled.reshape(1, time_steps, 1)

        yhat_scaled = model.predict(x_input, verbose=0)[0, 0]
        yhat = scaler.inverse_transform([[yhat_scaled]])[0, 0]

        preds.append(yhat)
        history.append(true_val)

    preds = np.array(preds)

    metrics = {
        "RMSE": rmse(test_vals.flatten(), preds),
        "MAE": mae(test_vals.flatten(), preds),
        "sMAPE%": smape(test_vals.flatten(), preds),
        "CORR": corr(test_vals.flatten(), preds),
        "TIME_STEPS": time_steps,
        "EPOCHS": epochs,
    }

    pred_series = pd.Series(preds, index=y.index[-test_horizon:])

    return pred_series, metrics

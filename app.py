import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data_utils import load_series_from_csv, split_series
from models.arima_model import fit_predict_evaluate as run_arima
from models.svr_model import fit_predict_evaluate as run_svr
from models.lstm_model import fit_predict_evaluate as run_lstm


# =========================
# Page config
# =========================
st.set_page_config(layout="wide")
st.title("Time Series Forecasting Model Comparison")
st.caption(
    "Comparison of ARIMA, SVR, and LSTM using rolling one-step forecasting "
    "on a time-based test split. No future forecasting is performed."
)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader(
        "Upload CSV (Date, Close)",
        type=["csv"]
    )

    st.divider()
    st.header("Test Split")
    test_horizon = st.number_input(
        "Test horizon",
        min_value=30,
        max_value=500,
        value=252,
        step=1
    )

    st.divider()
    st.header("ARIMA")
    p_max = st.slider("Max p", 0, 6, 3)
    q_max = st.slider("Max q", 0, 6, 3)
    trends = st.multiselect(
        "Trend candidates",
        ["n", "c", "t"],
        default=["n", "c", "t"]
    )

    st.divider()
    st.header("SVR (Linear)")
    lags = st.slider("Lags", 3, 120, 10)
    C = st.number_input("C", 0.01, 1000.0, 1.0)
    epsilon = st.number_input("Epsilon", 0.001, 10.0, 0.1)

    st.divider()
    st.header("LSTM")
    time_steps = st.slider("Time steps", 10, 120, 60)
    epochs = st.slider("Epochs", 5, 50, 20)

    st.divider()
    run_button = st.button("Run Evaluation", type="primary")


# =========================
# Guard: upload required
# =========================
if uploaded is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

# =========================
# Load & split data
# =========================
try:
    y = load_series_from_csv(uploaded)
    train, test = split_series(y, int(test_horizon))
except Exception as e:
    st.error(str(e))
    st.stop()


tabs = st.tabs(["Overview", "ARIMA", "SVR", "LSTM"])

# =========================
# Overview (before run)
# =========================
with tabs[0]:
    st.subheader("Series and split")
    fig = plt.figure(figsize=(12, 4))
    plt.plot(train.index, train.values, label="Train")
    plt.plot(test.index, test.values, label="Test")
    plt.axvline(test.index.min(), linestyle=":", color="black")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.write({
        "Total points": len(y),
        "Train points": len(train),
        "Test points": len(test),
        "Start date": str(y.index.min().date()),
        "End date": str(y.index.max().date())
    })

if not run_button:
    st.stop()

# =========================
# Run models
# =========================
with st.spinner("Running ARIMA..."):
    arima_pred, arima_metrics = run_arima(
        y=y,
        test_horizon=test_horizon,
        p_max=p_max,
        q_max=q_max,
        trends=trends
    )

with st.spinner("Running SVR..."):
    svr_pred, svr_metrics = run_svr(
        y=y,
        test_horizon=test_horizon,
        lags=lags,
        C=C,
        epsilon=epsilon
    )

with st.spinner("Running LSTM (training)..."):
    try:
        lstm_pred, lstm_metrics = run_lstm(
            y=y,
            test_horizon=test_horizon,
            time_steps=time_steps,
            epochs=epochs
        )
        lstm_error = None
    except Exception as e:
        lstm_pred = None
        lstm_metrics = None
        lstm_error = str(e)

# =========================
# Metrics table
# =========================
rows = [
    {"Model": "ARIMA", **arima_metrics},
    {"Model": "SVR", **svr_metrics},
]

if lstm_metrics is not None:
    rows.append({"Model": "LSTM", **lstm_metrics})
else:
    rows.append({"Model": "LSTM", "Error": lstm_error})

metrics_df = pd.DataFrame(rows)

with tabs[0]:
    st.subheader("Evaluation Metrics (Test Set)")
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Prediction comparison (test set)")
    fig = plt.figure(figsize=(12, 4))
    plt.plot(test.index, test.values, label="Actual")
    plt.plot(arima_pred.index, arima_pred.values, label="ARIMA")
    plt.plot(svr_pred.index, svr_pred.values, label="SVR")
    if lstm_pred is not None:
        plt.plot(lstm_pred.index, lstm_pred.values, label="LSTM")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

# =========================
# ARIMA tab
# =========================
with tabs[1]:
    st.subheader("ARIMA Results")
    st.write(arima_metrics)
    fig = plt.figure(figsize=(12, 4))
    plt.plot(test.index, test.values, label="Actual")
    plt.plot(arima_pred.index, arima_pred.values, label="ARIMA")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

# =========================
# SVR tab
# =========================
with tabs[2]:
    st.subheader("SVR Results")
    st.write(svr_metrics)
    fig = plt.figure(figsize=(12, 4))
    plt.plot(test.index, test.values, label="Actual")
    plt.plot(svr_pred.index, svr_pred.values, label="SVR")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

# =========================
# LSTM tab
# =========================
with tabs[3]:
    st.subheader("LSTM Results")
    if lstm_metrics is None:
        st.error(lstm_error)
    else:
        st.write(lstm_metrics)
        fig = plt.figure(figsize=(12, 4))
        plt.plot(test.index, test.values, label="Actual")
        plt.plot(lstm_pred.index, lstm_pred.values, label="LSTM")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)

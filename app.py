import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data_utils import load_series_from_csv, split_series
from models.arima_model import fit_predict_evaluate as run_arima
from models.svr_model import fit_predict_evaluate as run_svr


st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Time Series Model Comparison")
st.caption("Evaluation uses rolling one-step prediction on the test split (time-based). No future forecasting is performed.")


with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    st.divider()
    st.header("Split")
    test_horizon = st.number_input(
        "Test horizon (business days)",
        min_value=30,
        max_value=500,
        value=252,
        step=1
    )

    st.divider()
    st.header("ARIMA (AIC grid search)")
    p_max = st.slider("Max p", 0, 6, 3)
    q_max = st.slider("Max q", 0, 6, 3)
    trends = st.multiselect("Trend candidates", ["n", "c", "t"], default=["n", "c", "t"])

    st.divider()
    st.header("SVR (linear)")
    lags = st.slider("Lags", 3, 120, 10)
    C = st.number_input("C", min_value=0.01, max_value=1000.0, value=1.0, step=0.1)
    epsilon = st.number_input("Epsilon", min_value=0.001, max_value=10.0, value=0.1, step=0.01)

    st.divider()
    run_button = st.button("Run evaluation", type="primary")


if uploaded is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

# Load + clean
try:
    y = load_series_from_csv(uploaded)
except Exception as e:
    st.error(f"Failed to load/clean data: {e}")
    st.stop()

# Split
try:
    train, test = split_series(y, int(test_horizon))
except Exception as e:
    st.error(f"Failed to split series: {e}")
    st.stop()

tab_overview, tab_arima, tab_svr = st.tabs(["Overview", "ARIMA", "SVR"])

with tab_overview:
    st.subheader("Series and split")
    fig = plt.figure(figsize=(12, 4))
    plt.plot(train.index, train.values, label="Train")
    plt.plot(test.index, test.values, label="Test")
    plt.axvline(x=test.index.min(), linestyle=":", linewidth=2, label="Split")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.write(
        {
            "Total points": int(len(y)),
            "Train points": int(len(train)),
            "Test points": int(len(test)),
            "Start": str(y.index.min().date()),
            "End": str(y.index.max().date()),
        }
    )

if not run_button:
    st.info("Configure parameters in the sidebar, then click Run evaluation.")
    st.stop()

# Run ARIMA
with st.spinner("Running ARIMA..."):
    arima_pred, arima_metrics = run_arima(
        y=y,
        test_horizon=int(test_horizon),
        p_max=int(p_max),
        q_max=int(q_max),
        trends=list(trends),
    )

# Run SVR
with st.spinner("Running SVR..."):
    svr_pred, svr_metrics = run_svr(
        y=y,
        test_horizon=int(test_horizon),
        lags=int(lags),
        C=float(C),
        epsilon=float(epsilon),
    )

# Metrics table
rows = [
    {"Model": "ARIMA", **{k: v for k, v in arima_metrics.items() if k != "ORDER"}},
    {"Model": "SVR (linear)", **svr_metrics},
]
metrics_df = pd.DataFrame(rows)

preferred_cols = ["Model", "RMSE", "MAE", "sMAPE%", "CORR"]
other_cols = [c for c in metrics_df.columns if c not in preferred_cols]
metrics_df = metrics_df[preferred_cols + other_cols]

with tab_overview:
    st.subheader("Metrics (test set)")
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Test set prediction comparison")
    fig = plt.figure(figsize=(12, 4))
    plt.plot(test.index, test.values, label="Actual (test)")
    plt.plot(arima_pred.index, arima_pred.values, label="ARIMA (rolling 1-step)")
    plt.plot(svr_pred.index, svr_pred.values, label="SVR (rolling 1-step)")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

with tab_arima:
    st.subheader("ARIMA details")
    st.write(arima_metrics)
    fig = plt.figure(figsize=(12, 4))
    plt.plot(test.index, test.values, label="Actual (test)")
    plt.plot(arima_pred.index, arima_pred.values, label="ARIMA (rolling 1-step)")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

with tab_svr:
    st.subheader("SVR details")
    st.write(svr_metrics)
    fig = plt.figure(figsize=(12, 4))
    plt.plot(test.index, test.values, label="Actual (test)")
    plt.plot(svr_pred.index, svr_pred.values, label="SVR (rolling 1-step)")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

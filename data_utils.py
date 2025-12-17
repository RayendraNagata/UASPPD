import numpy as np
import pandas as pd


DATE_CANDIDATES = ["Date", "date", "Datetime", "datetime", "Timestamp", "timestamp"]
VALUE_CANDIDATES = ["Close", "close", "Adj Close", "adj_close", "Price", "price", "Value", "value"]


def pick_date_column(df: pd.DataFrame) -> str:
    for c in DATE_CANDIDATES:
        if c in df.columns:
            return c
    # fallback: first non-numeric column
    for c in df.columns:
        if not np.issubdtype(df[c].dtype, np.number):
            return c
    return df.columns[0]


def pick_value_column(df: pd.DataFrame) -> str:
    for c in VALUE_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        "Value column not found. Expected one of: "
        + ", ".join(VALUE_CANDIDATES)
        + f". Found: {list(df.columns)}"
    )


def load_series_from_csv(uploaded_file) -> pd.Series:
    raw = pd.read_csv(uploaded_file)
    date_col = pick_date_column(raw)
    val_col = pick_value_column(raw)

    df = raw.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df = df.drop_duplicates(subset=[date_col], keep="last")
    df = df[[date_col, val_col]].rename(columns={date_col: "Date", val_col: "Value"}).set_index("Date")
    df = df.sort_index()

    # Business-day index + fill
    idx = pd.date_range(df.index.min(), df.index.max(), freq="B")
    df = df.reindex(idx)
    df["Value"] = df["Value"].ffill().bfill()

    s = df["Value"].astype(float)
    if len(s) < 60:
        raise ValueError("Series too short after cleaning. Provide more data.")
    if s.isna().any():
        raise ValueError("NaNs remain after cleaning. Check date/value parsing.")
    return s


def split_series(y: pd.Series, test_horizon: int) -> tuple[pd.Series, pd.Series]:
    if test_horizon >= len(y):
        raise ValueError("test_horizon is larger than series length.")
    train = y.iloc[:-test_horizon]
    test = y.iloc[-test_horizon:]
    return train, test

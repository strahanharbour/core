import pandas as pd

from main.data.pipelines.transformer import DataTransformer


def test_transformer_drops_columns_and_stringifies_date():
    df = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.5, 2.5],
            "Low": [0.5, 1.5],
            "Close": [1.2, 2.2],
            "Adj Close": [1.1, 2.1],
            "Volume": [100, 200],
            "Dividends": [0.0, 0.0],
            "Stock Splits": [0.0, 0.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    out = DataTransformer().clean_data(df)
    assert "Dividends" not in out.columns
    assert "Stock Splits" not in out.columns and "Splits" not in out.columns
    assert "date" in out.columns
    assert out["date"].dtype == object
    assert list(out["date"]) == ["2024-01-01", "2024-01-02"]

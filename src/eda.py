import pandas as pd


def data_load_and_summary(file):
    df = pd.read_csv("../data/raw/complaints.csv")
    summary = {
        "total_rows": len(df),
        "columns": df.columns.tolist(),
        "dataset shape": df.shape,
        "missing values": df.isnull()

    }
    return df, summary

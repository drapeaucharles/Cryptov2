import pandas as pd

def load_data():
    def clean(df):
        df = df.select_dtypes(include=[np.number])
        return df
    return {
        "15m": clean(pd.read_csv("data/btc_15m_features.csv")),
        "1h": clean(pd.read_csv("data/btc_1h_features.csv")),
        "2h": clean(pd.read_csv("data/btc_2h_features.csv")),
        "4h": clean(pd.read_csv("data/btc_4h_features.csv")),
    }


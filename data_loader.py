import pandas as pd
import numpy as np

def load_data():
    def clean(df):
        df = df.select_dtypes(include=[np.number])  # Keep only numeric columns
        return df

    df_15m = clean(pd.read_csv('data/btc_15m_features.csv'))
    df_1h = clean(pd.read_csv('data/btc_1h_features.csv'))
    df_2h = clean(pd.read_csv('data/btc_2h_features.csv'))
    df_4h = clean(pd.read_csv('data/btc_4h_features.csv'))

    return {'15m': df_15m, '1h': df_1h, '2h': df_2h, '4h': df_4h}

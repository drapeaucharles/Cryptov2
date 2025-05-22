import pandas as pd

def load_data():
    df_15m = pd.read_csv('data/btc_15m_features.csv')
    df_1h = pd.read_csv('data/btc_1h_features.csv')
    df_2h = pd.read_csv('data/btc_2h_features.csv')
    df_4h = pd.read_csv('data/btc_4h_features.csv')
    return {'15m': df_15m, '1h': df_1h, '2h': df_2h, '4h': df_4h}

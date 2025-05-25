# Placeholder for analyzing trade logs
# After logging trades in training, use this script to compute metrics:
# - Winrate
# - Average SL/TP
# - Profit factor
# - Avg trade duration
# - Trade distribution

import pandas as pd

def analyze_trades(log_file='logs/trades.csv'):
    df = pd.read_csv(log_file)

    total_trades = len(df)
    win_trades = df[df['PnL'] > 0]
    lose_trades = df[df['PnL'] <= 0]
    avg_sl = df['SL'].mean()
    avg_tp = df['TP'].mean()

    print(f"Total trades: {total_trades}")
    print(f"Win rate: {len(win_trades) / total_trades * 100:.2f}%")
    print(f"Average SL: {avg_sl:.4f}, Average TP: {avg_tp:.4f}")
    profit_factor = win_trades['PnL'].sum() / abs(lose_trades['PnL'].sum()) if not lose_trades.empty else float('inf')
    print(f"Profit Factor: {profit_factor:.2f}")

if __name__ == '__main__':
    analyze_trades()

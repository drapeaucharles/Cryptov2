import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import csv
import os

class AdvancedTradingEnv(gym.Env):
    def __init__(self, df_dict, initial_balance=1000, max_concurrent_trades=3, log_path="logs/trades.csv"):
        super(AdvancedTradingEnv, self).__init__()

        self.df_dict = df_dict  # Dict of synced dataframes for 15m, 1h, 2h, 4h
        self.initial_balance = initial_balance
        self.max_trades = max_concurrent_trades

        self.fee = 0.0004  # 0.04%
        self.leverage = 10
        self.max_risk = 0.01  # Max 1% of capital per trade

        self.min_sl = 0.002  # 0.2%
        self.max_sl = 0.05   # 5%

        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'EntryPrice', 'SL', 'TP', 'ExitPrice', 'PnL', 'Reason'])

        self.action_space = spaces.Dict({
            "action": spaces.Discrete(3),  # 0 = Hold, 1 = Buy, 2 = Sell
            "sl": spaces.Box(self.min_sl, self.max_sl, shape=(1,), dtype=np.float32),
            "tp": spaces.Box(self.min_sl * 2, self.max_sl * 3, shape=(1,), dtype=np.float32)
        })

        obs_len = sum([df.shape[1] for df in df_dict.values()])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.trades = []
        self.current_step = 100  # Start after indicators have enough data
        self.resting = 0
        self.sl_count = 0

        return self._next_observation(), {}

    def _next_observation(self):
        obs = []
        for df in self.df_dict.values():
            obs.extend(df.iloc[self.current_step].values)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        done = False
        reward = 0

        if self.resting > 0:
            self.resting -= 1
            self.current_step += 1
            return self._next_observation(), 0, False, {}

        a_type = action['action']
        sl = float(action['sl'])
        tp = float(action['tp'])

        price = self.df_dict['15m'].iloc[self.current_step]['close']

        if a_type in [1, 2] and len(self.trades) < self.max_trades:
            risk_amount = self.balance * self.max_risk
            position_size = (risk_amount * self.leverage) / (sl * price)
            direction = 1 if a_type == 1 else -1

            self.trades.append({
                "entry": price,
                "size": position_size,
                "sl": price * (1 - direction * sl),
                "tp": price * (1 + direction * tp),
                "dir": direction,
                "open_step": self.current_step,
                "sl_val": sl,
                "tp_val": tp
            })

        closed_trades = []
        for trade in self.trades:
            high = self.df_dict['15m'].iloc[self.current_step]['high']
            low = self.df_dict['15m'].iloc[self.current_step]['low']

            hit_sl = (trade['dir'] == 1 and low <= trade['sl']) or (trade['dir'] == -1 and high >= trade['sl'])
            hit_tp = (trade['dir'] == 1 and high >= trade['tp']) or (trade['dir'] == -1 and low <= trade['tp'])

            exit_reason = None

            if hit_tp:
                exit_price = trade['tp']
                pnl = (exit_price - trade['entry']) * trade['dir'] * trade['size']
                exit_reason = 'TP'
            elif hit_sl:
                exit_price = trade['sl']
                pnl = (exit_price - trade['entry']) * trade['dir'] * trade['size']
                exit_reason = 'SL'
                self.sl_count += 1
            else:
                continue

            fee = exit_price * trade['size'] * self.fee
            pnl -= fee
            reward += pnl
            self.balance += pnl
            closed_trades.append(trade)

            with open(self.log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade['open_step'], trade['entry'], trade['sl_val'], trade['tp_val'],
                    exit_price, pnl, exit_reason
                ])

        for trade in closed_trades:
            self.trades.remove(trade)

        if self.sl_count >= 5:
            self.resting = 96
            self.sl_count = 0

        self.net_worth = self.balance + sum(
            [(self.df_dict['15m'].iloc[self.current_step]['close'] - t['entry']) * t['dir'] * t['size']
             for t in self.trades]
        )

        self.current_step += 1
        if self.current_step >= len(self.df_dict['15m']) - 1:
            done = True

        return self._next_observation(), reward, done, {}

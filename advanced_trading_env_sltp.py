import gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import csv
import os

class AdvancedTradingEnv(gym.Env):
    def __init__(self, df_dict, initial_balance=1000, max_concurrent_trades=3, log_path="logs/trades.csv"):
        super(AdvancedTradingEnv, self).__init__()

        self.df_dict = df_dict
        for key in self.df_dict:
            self.df_dict[key] = self.df_dict[key].select_dtypes(include=[np.number])

        self.initial_balance = initial_balance
        self.max_trades = max_concurrent_trades

        self.fee = 0.0004
        self.leverage = 10
        self.max_risk = 0.01

        self.min_sl = 0.002
        self.max_sl = 0.05

        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'EntryPrice', 'SL', 'TP', 'ExitPrice', 'PnL', 'Reason'])

        obs_len = sum([df.shape[1] for df in df_dict.values()])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.trades = []
        self.current_step = 100
        self.resting = 0
        self.sl_count = 0

        obs = self._next_observation()
        return obs, {}

    def _next_observation(self):
        obs = []
        for df in self.df_dict.values():
            row = df.iloc[self.current_step]
            obs.extend(row.values)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        done = False
        reward = 0

        if self.resting > 0:
            self.resting -= 1
            self.current_step += 1
            obs = self._next_observation()
            return obs, 0, False, {}

        a_type = int(round((action[0] + 1) * 1.0))
        sl = ((action[1] + 1) / 2) * (self.max_sl - self.min_sl) + self.min_sl
        tp = ((action[2] + 1) / 2) * (self.max_sl * 3 - self.min_sl * 2) + self.min_sl * 2

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

        obs = self._next_observation()
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, {}

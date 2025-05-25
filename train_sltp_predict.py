import os
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from multi_output_policy import CustomMultiOutputPolicy
from advanced_trading_env_sltp import AdvancedTradingEnv

def load_data():
    df_15m = pd.read_csv('data/btc_15m_features.csv')
    df_1h = pd.read_csv('data/btc_1h_features.csv')
    df_2h = pd.read_csv('data/btc_2h_features.csv')
    df_4h = pd.read_csv('data/btc_4h_features.csv')
    return {'15m': df_15m, '1h': df_1h, '2h': df_2h, '4h': df_4h}

def main():
    df_dict = load_data()
    env = DummyVecEnv([lambda: Monitor(AdvancedTradingEnv(df_dict, log_path="logs/trades.csv"))])

    model = PPO(
        policy=CustomMultiOutputPolicy,
        env=env,
        verbose=1,
        tensorboard_log="./logs/",
        device='cuda' if torch.cuda.is_available() else 'cpu',
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000,
        save_path="./models/",
        name_prefix="ppo_trading"
    )

    model.learn(total_timesteps=100_000_000, callback=checkpoint_callback)
    model.save("./models/final_model")

if __name__ == "__main__":
    main()

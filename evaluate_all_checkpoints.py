import os
import torch
import pandas as pd
from stable_baselines3 import PPO
from multi_output_policy import CustomMultiOutputPolicy
from advanced_trading_env_sltp import AdvancedTradingEnv

def load_data():
    df_15m = pd.read_csv('data/BTCUSDT_15m.csv')
    df_1h = pd.read_csv('data/BTCUSDT_1h.csv')
    df_2h = pd.read_csv('data/BTCUSDT_2h.csv')
    df_4h = pd.read_csv('data/BTCUSDT_4h.csv')
    return {'15m': df_15m, '1h': df_1h, '2h': df_2h, '4h': df_4h}

def evaluate_model(model_path):
    df_dict = load_data()
    env = AdvancedTradingEnv(df_dict)
    model = PPO.load(model_path, env=env, custom_objects={"policy_class": CustomMultiOutputPolicy})

    obs = env.reset()
    done = False
    net_worths = []
    trades_count = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step({
            'action': action,
            'sl': torch.tensor([0.01]),
            'tp': torch.tensor([0.03])
        })
        net_worths.append(env.net_worth)
        trades_count = len(env.trades)

    print(f"Checkpoint: {model_path}")
    print(f"Final Net Worth: {env.net_worth:.2f}")
    print(f"Trades Open: {trades_count}")
    print(f"Drawdown: {max(net_worths) - min(net_worths):.2f}\n")

def main():
    checkpoints_dir = "models/"
    for file in sorted(os.listdir(checkpoints_dir)):
        if file.endswith(".zip") and file.startswith("ppo_trading"):
            evaluate_model(os.path.join(checkpoints_dir, file))

if __name__ == "__main__":
    main()

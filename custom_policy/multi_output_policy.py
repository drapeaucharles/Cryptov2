import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class MultiOutputExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.net(observations)


class CustomMultiOutputPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMultiOutputPolicy, self).__init__(
            *args,
            features_extractor_class=MultiOutputExtractor,
            **kwargs
        )

        # Replace action_net to produce 3 continuous outputs: action_type (0–2), SL, TP
        last_layer_dim = self.mlp_extractor.policy_net[-1].out_features
        self.action_net = nn.Sequential(
            nn.Linear(last_layer_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()  # Outputs in [0, 1] range — will be scaled by environment
        )

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        action_logits = self.action_net(latent_pi)
        actions = action_logits  # Continuous output
        values = self.value_net(latent_vf)
        log_prob = None  # PPO will handle

        return actions, values, log_prob

    def _predict(self, obs, deterministic=False):
        actions, _, _ = self.forward(obs, deterministic)
        return actions

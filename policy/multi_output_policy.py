from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
import torch


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

        # Override action_net to produce multi-output head
        self.action_logits = nn.Linear(self.mlp_extractor.policy_net[-1].out_features, 3)  # Discrete: Buy/Sell/Hold
        self.sl_output = nn.Sequential(
            nn.Linear(self.mlp_extractor.policy_net[-1].out_features, 1),
            nn.Sigmoid()  # Output between 0 and 1, scale later
        )
        self.tp_output = nn.Sequential(
            nn.Linear(self.mlp_extractor.policy_net[-1].out_features, 1),
            nn.Sigmoid()
        )

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        action_logits = self.action_logits(latent_pi)
        sl_raw = self.sl_output(latent_pi)
        tp_raw = self.tp_output(latent_pi)

        distribution = self._get_action_dist_from_latent(action_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, log_prob, self.value_net(latent_vf), {'sl': sl_raw, 'tp': tp_raw}

    def _predict(self, obs, deterministic=False):
        actions, _, _, _ = self.forward(obs, deterministic)
        return actions

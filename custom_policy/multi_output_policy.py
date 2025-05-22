import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution


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

        self.action_dist = DiagGaussianDistribution(self.action_space.shape[0])

        last_layer_dim = self.mlp_extractor.latent_dim_pi
        self.mu = nn.Linear(last_layer_dim, self.action_space.shape[0])
        self.log_std = nn.Parameter(torch.ones(self.action_space.shape[0]) * -0.5)

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        mean_actions = self.mu(latent_pi)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def _predict(self, obs, deterministic=False):
        actions, _, _ = self.forward(obs, deterministic)
        return actions

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


MODEL_LOADERS = {
    "a2c": A2C,
    "ddpg": DDPG,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}


def normalize_algo_name(algo: str) -> str:
    key = algo.lower().strip()
    if key not in MODEL_LOADERS:
        raise ValueError(f"Unsupported Stable-Baselines3 algorithm: {algo}")
    return key


def load_sb3_model(algo: str, model_path, env=None):
    key = normalize_algo_name(algo)
    model_cls = MODEL_LOADERS[key]
    return model_cls.load(str(Path(model_path).resolve()), env=env)


class _SpaceOnlyEnv(gym.Env):
    metadata = {}

    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def _zero_obs(self):
        if isinstance(self.observation_space, spaces.Box):
            return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        raise TypeError("Only Box observation spaces are supported for VecNormalize inference.")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._zero_obs(), {}

    def step(self, action):
        return self._zero_obs(), 0.0, True, False, {}


def load_vecnormalize_for_inference(vecnormalize_path, observation_space, action_space):
    if vecnormalize_path is None:
        return None

    path = Path(vecnormalize_path).resolve()
    if not path.exists():
        return None

    dummy_vec_env = DummyVecEnv([lambda: _SpaceOnlyEnv(observation_space, action_space)])
    vecnormalize = VecNormalize.load(str(path), dummy_vec_env)
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    return vecnormalize


class NormalizedModelPredictor:
    def __init__(self, model, vecnormalize):
        self.model = model
        self.vecnormalize = vecnormalize
        self.observation_space = model.observation_space
        self.action_space = model.action_space

    def predict(self, observation, *args, **kwargs):
        obs_array = np.asarray(observation, dtype=np.float32)
        normalized_obs = self.vecnormalize.normalize_obs(obs_array)
        return self.model.predict(normalized_obs, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


def load_sb3_model_for_inference(algo: str, model_path, vecnormalize_path=None):
    model = load_sb3_model(algo, model_path)
    vecnormalize = load_vecnormalize_for_inference(
        vecnormalize_path,
        observation_space=model.observation_space,
        action_space=model.action_space,
    )
    if vecnormalize is None:
        return model
    return NormalizedModelPredictor(model, vecnormalize)

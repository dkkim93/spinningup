import gym
import gym_env
from spinup import ppo_pytorch as ppo

ppo(
    env_fn=lambda: gym.make('pointmass-v0'), 
    ac_kwargs=dict(hidden_sizes=[16] * 2),
    gamma=0.99,
    max_ep_len=1000,
    lam=0.95,
    epochs=100000,
    seed=1)

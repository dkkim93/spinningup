import gym
import gym_env
from spinup import ppo_pytorch as ppo
from gym_env.wrapper import PendulumCostWrapper

env = gym.make('Pendulum-v0')
env._max_episode_steps = 100
env = PendulumCostWrapper(env)

ppo(
    env_fn=lambda: env, 
    ac_kwargs=dict(hidden_sizes=[16] * 2),
    gamma=0.99,
    max_ep_len=1000,
    lam=0.95,
    epochs=100000,
    seed=1)

from gym.envs.registration import register


########################################################################################
# POINTMASS
register(
    id='pointmass-v0',
    entry_point='gym_env.pointmass.pointmass_env:PointMassEnv',
    max_episode_steps=100
)

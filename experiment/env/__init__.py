
#能够通过标准gym.make搞出来

from gym.envs.registration import register

register(
    id='LongHorizon-v0',
    entry_point='test.env.longH:LongHorizonEnv',
    max_episode_steps=50,
)


from gym.envs.registration import register



#cutom env
register(
        id='Kukagrasp-v0',
    entry_point='/home/wyq/SW/long_horizon_chainning/SIM_env/sim_env/kuka_goal_env/kuka_slsc_wrapper.py:KukaGraspEnv',
    max_episode_steps=26,
)
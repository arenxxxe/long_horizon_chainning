from ray.rllib.algorithms.ppo import PPOConfig
from chaining_package.ENV.agent_interface_env.kuka_slsc_wrapper import KukagraspSLWrapper
from chaining_package.ENV.task_env.kuka_grasp_env import KukaGraspEnv


config = (
    PPOConfig().environment(
        # Env class to use (here: our gym.Env sub-class from above).
        env=KukaGraspEnv,
        # Config dict to be passed to our custom env's constructor.
        # Use corridor with 20 fields (including S and G).
        env_config={"corridor_length": 28},
    )
    # Parallelize environment rollouts.
    .env_runners(num_env_runners=3)
)






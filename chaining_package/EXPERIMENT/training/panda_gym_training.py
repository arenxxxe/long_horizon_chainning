import gymnasium as gym
import panda_gym

import logging
from stable_baselines3 import DDPG


# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_step_data(observation, reward, terminated, truncated, info):
    logging.info(f"Observation: {observation}")
    logging.info(f"Reward: {reward}")
    logging.info(f"Terminated: {terminated}")
    logging.info(f"Truncated: {truncated}")
    logging.info(f"Info: {info}")
import time
def main():

    env = gym.make("PandaPickAndPlaceDense-v3", render_mode="human")
    # for i in range(100):
        
    #     observation, info = env.reset()
    #     model = DDPG(policy="MultiInputPolicy", env=env)
    #     model.load("./hello.pth")
    #     model.learn(30_000,log_interval=1)
    #     model.save("./hello.pth")
    #     model.logger.log()

    env.reset()

    for _ in range(1000):

        action = env.action_space.sample() # random action
        breakpoint()
        observation, reward, terminated, truncated, info = env.step(action)
        log_step_data(observation, reward, terminated, truncated, info)
        time.sleep(1/30)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
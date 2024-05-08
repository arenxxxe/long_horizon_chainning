import gynasium as gym
import time
import numpy as np
import wandb 

class LongHorizonEnv(gym.Env):
        def __init__(self, render_mode: str = None):
                pass

        def step(self, action: np.ndarray):
                pass

        def reset(self):
                pass

        def close(self):
                pass

        def render(self, mode='rgb_array'):
                pass

        def seed(self, seed=None):
                pass

        def compute_reward(self, achieved_goal, desired_goal, info):
                pass

        def _env_setup(self):
                pass

        def _get_obs(self):
                pass

        def _set_action(self, action):
                pass

        def _is_success(self, achieved_goal, desired_goal):
                pass

        def _sample_goal(self):
                pass

        def _sample_goal_callback(self):
                pass

        def _render_callback(self, mode):
                pass

        def _step_callback(self):
                pass


        def test(self, horizon=100):
                
                steps, done = 0, False
                obs = self.reset()
                while not done and steps <= horizon:
                tic = time.time()
                action = self.get_oracle_action(obs)
                print('\n -> step: {}, action: {}'.format(steps, np.round(action, 4)))
                # print('action:', action)
                obs, reward, done, info = self.step(action)
                if isinstance(obs, dict):
                        print(" -> achieved goal: {}".format(np.round(obs['achieved_goal'], 4)))
                        print(" -> desired goal: {}".format(np.round(obs['desired_goal'], 4)))
                else:
                        print(" -> achieved goal: {}".format(np.round(info['achieved_goal'], 4)))
                done = info['is_success'] if isinstance(obs, dict) else done
                steps += 1
                toc = time.time()
                print(" -> step time: {:.4f}".format(toc - tic))
                time.sleep(0.05)
                print('\n -> Done: {}\n'.format(done > 0))
                


class SurRoLGoalEnv(LongHorizonEnv):
    """
    A gym GoalEnv wrapper for SurRoL.
    refer to: https://github.com/openai/gym/blob/master/gym/core.py
    """

    def reset(self):
        # Enforce that each GoalEnv uses a Goal-compatible observation space.
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise error.Error('GoalEnv requires an observation space of type gym.spaces.Dict')
        for key in ['observation', 'achieved_goal', 'desired_goal']:
            if key not in self.observation_space.spaces:
                raise error.Error('GoalEnv requires the "{}" key to be part of the observation dictionary.'.format(key))
        return super().reset()
    


if __name__ == "__main__":
        env = LongHorizonEnv()
        env.test(horizon=100)
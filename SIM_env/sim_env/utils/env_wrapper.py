import gym
import copy
import numpy as np

from surrol.utils.pybullet_utils import get_link_pose, pairwise_collision
import abc
from contextlib import contextmanager


class SkillChainingWrapper(gym.Wrapper):
    def __init__(self, env, subtask, output_raw_obs):
        super().__init__(env)
        self.subtask = subtask
        self._start_subtask = subtask
        self._elapsed_steps = None
        self._output_raw_obs = output_raw_obs

    @abc.abstractmethod
    def _replace_goal_with_subgoal(self, obs):
        """Replace achieved goal and desired goal."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def _subgoal(self):
        """Output goal of subtask."""
        raise NotImplementedError

    @contextmanager
    def switch_subtask(self):
        self.subtask = self.SUBTASK_PREV_SUBTASK[self.subtask]
        yield
        self.subtask = self.SUBTASK_NEXT_SUBTASK[self.subtask]


class PrimitiveWrapper(SkillChainingWrapper):
    '''Wrapper for skill learning'''
    SUBTASK_ORDER = {
        'grasp': 0,
        'handover': 1,
        'release': 2
    }    
    SUBTASK_STEPS = {
        'grasp': 45,
        'handover': 35,
        'release': 20
    }
    SUBTASK_RESET_INDEX = {
        'handover': 4,
        'release': 10
    }
    SUBTASK_RESET_MAX_STEPS = {
        'handover': 60,
        'release': 90
    }
    SUBTASK_PREV_SUBTASK = {
        'handover': 'grasp',
        'release': 'handover'
    }
    SUBTASK_NEXT_SUBTASK = {
        'grasp': 'handover',
        'handover': 'release'
    }
    def __init__(self, env, subtask='grasp', output_raw_obs=False):
        super().__init__(env, subtask, output_raw_obs)
        self.done_subtasks = {key: False for key in self.SUBTASK_STEPS.keys()}

    @property
    def max_episode_steps(self):
        assert np.sum([x for x in self.SUBTASK_STEPS.values()]) == self.env._max_episode_steps
        return self.SUBTASK_STEPS[self.subtask]

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        next_obs_ = self._replace_goal_with_subgoal(next_obs.copy())
        reward = self.compute_reward(next_obs_['achieved_goal'], next_obs_['desired_goal'])
        info['is_success'] = reward + 1
        done = self._elapsed_steps == self.max_episode_steps
        self._elapsed_steps += 1

        if self._output_raw_obs: return next_obs_, reward, done, info, next_obs
        else: return next_obs_, reward, done, info,

    def reset(self):
        self.subtask = self._start_subtask
        if self.subtask not in self.SUBTASK_RESET_INDEX.keys():
            obs = self.env.reset() 
            self._elapsed_steps = 0 
        else:
            success = False
            while not success:
                obs = self.env.reset() 
                self.subtask = self._start_subtask
                self._elapsed_steps = 0 

                action, skill_index = self.env.get_oracle_action(obs)
                count, max_steps = 0, self.SUBTASK_RESET_MAX_STEPS[self.subtask]
                while skill_index < self.SUBTASK_RESET_INDEX[self.subtask] and count < max_steps:
                    obs, reward, done, info = self.env.step(action)
                    action, skill_index = self.env.get_oracle_action(obs)
                    count += 1

                # Reset again if failed
                with self.switch_subtask():
                    obs_ = self._replace_goal_with_subgoal(obs.copy())
                    success = self.compute_reward(obs_['achieved_goal'], obs_['desired_goal']) + 1

        if self._output_raw_obs: return self._replace_goal_with_subgoal(obs), obs
        else: return self._replace_goal_with_subgoal(obs)

    #---------------------------Observation---------------------------
    def _replace_goal_with_subgoal(self, obs):
        """Replace ag and g"""
        subgoal = self._subgoal()    
        psm1col = pairwise_collision(self.env.obj_id, self.psm1.body)
        psm2col = pairwise_collision(self.env.obj_id, self.psm2.body)
        pos_obj1, _ = get_link_pose(self.env.obj_id, self.env.obj_link1)

        if self.subtask == 'grasp':
            obs['achieved_goal'] = np.append(obs['achieved_goal'], [psm1col, psm2col])
            obs['desired_goal'] = np.append(subgoal, [0, 1])
        elif self.subtask == 'handover':
            obs['achieved_goal'] = np.append(pos_obj1, [psm1col, psm2col])
            obs['desired_goal'] = np.append(subgoal, [1, 0])
        elif self.subtask == 'release':
            obs['achieved_goal'] = np.append(obs['achieved_goal'], [psm1col, psm2col])
            obs['desired_goal'] = np.append(subgoal, [0, 0])
        return obs

    def _subgoal(self):
        """Output goal of subtask"""
        goal = self.env.subgoals[self.SUBTASK_ORDER[self.subtask]]
        return goal

    #---------------------------Reward---------------------------
    def compute_reward(self, ag, g, info=None):
        """Compute reward that indicates the success of subtask"""
        if len(ag.shape) == 1:
            goal_reach = self.env.compute_reward(ag[:3], g[:3], None) + 1
            contact_cond = np.all(ag[3:]==g[3:])
            if self.subtask == 'handover':
                reward = (goal_reach and contact_cond) - 1
            else:
                reward = goal_reach - 1
        else:
            goal_reach = self.env.compute_reward(ag[:,:3], g[:,:3], None).reshape(-1, 1) + 1
            if self.subtask == 'handover':
                contact_cond = np.all(ag[:, 3:]==g[:, 3:], axis=1).reshape(-1, 1)
                reward = np.all(np.hstack([goal_reach, contact_cond]), axis=1) - 1.
            else:
                reward = goal_reach - 1
        return reward
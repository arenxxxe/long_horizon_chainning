import abc
from contextlib import contextmanager

import gym
import numpy as np
import torch
from surrol.utils.pybullet_utils import (pairwise_collision,
                                         pairwise_link_collision)


def approx_collision(goal_a, goal_b, th=0.025):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1) < th

#子任务技能训练的部分  不需要任何的改变 
class SkillLearningWrapper(gym.Wrapper):
    #不需要修改
    def __init__(self, env, subtask, output_raw_obs):
        super().__init__(env)
        self.subtask = subtask
        self._start_subtask = subtask
        self._elapsed_steps = None
        self._output_raw_obs = output_raw_obs
    #不需要修改
    @abc.abstractmethod
    def _replace_goal_with_subgoal(self, obs):
        """Replace achieved goal and desired goal."""
        raise NotImplementedError
    #不需要修改
    @abc.abstractmethod
    def _subgoal(self):
        """Output goal of subtask."""
        raise NotImplementedError

    @contextmanager
    def switch_subtask(self, subtask=None):
        '''Temporally switch subtask, default: next subtask'''
        if subtask is not None:
            curr_subtask = self.subtask            
            self.subtask = subtask
            yield
            self.subtask = curr_subtask
        else:
            self.subtask = self.SUBTASK_PREV_SUBTASK[self.subtask]
            yield
            self.subtask = self.SUBTASK_NEXT_SUBTASK[self.subtask]


#-----------------------------Kukagrasp-v0-v0-----------------------------
class KukagraspSLWrapper(SkillLearningWrapper):
    '''Wrapper for skill learning'''
    #子任务顺序--完成
    SUBTASK_ORDER = {
        'grasp': 0,

        'release':1 
    }    
    #子任务步长env step的步长吧--完成
    SUBTASK_STEPS = {
        'grasp': 13,

        'release': 13
    }
    #与reset有关 源代码逻辑是除了开始的任务 后面的任务都写进来  不管了--完成
    SUBTASK_RESET_INDEX = {

        'release': 3 #第四个路点
    }
    
    #重置到一个子任务开始状态 环境首先需要执行多少步--完成
    SUBTASK_RESET_MAX_STEPS = {

        'release': 13
    }
    #对之前任务的了解-完成
    SUBTASK_PREV_SUBTASK = {

        'release': 'grasp'
    }
    #对下一个任务的了解--完成
    SUBTASK_NEXT_SUBTASK = {
        'grasp': 'release'

    }
    #子任务的接触条件--完成--
    #根据源代码  认为是在这个动作中 要不要接触物体的一种指引  如果你机械臂在这个动作中 需要最终抓到物体 那就是1 不要 那就是0
    SUBTASK_CONTACT_CONDITION = {
        'grasp': [1],

        'release': [ 0]
    }
    #最后一个子任务--完成
    LAST_SUBTASK = 'release'
    def __init__(self, env, subtask='grasp', output_raw_obs=False):
        super().__init__(env, subtask, output_raw_obs)
        self.done_subtasks = {key: False for key in self.SUBTASK_STEPS.keys()}
    #env设置最大的episode step-怀疑是register方法设置的--需要测试
    @property
    def max_episode_steps(self):
        assert np.sum([x for x in self.SUBTASK_STEPS.values()]) == self.env._max_episode_steps
        return self.SUBTASK_STEPS[self.subtask]
    #TODO 施工中  就是那两个接口
    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        next_obs_ = self._replace_goal_with_subgoal(next_obs.copy())
        reward = self.compute_reward(next_obs_['achieved_goal'], next_obs_['desired_goal'])
        info['is_success'] = reward + 1
        done = self._elapsed_steps == self.max_episode_steps
        # save groud truth goal
        with self.switch_subtask(self.LAST_SUBTASK):
            info['gt_goal'] = self._replace_goal_with_subgoal(next_obs.copy())['desired_goal']

        if self._output_raw_obs: return next_obs_, reward, done, info, next_obs
        else: return next_obs_, reward, done, info,
    #TODO 施工中
    def reset(self):
        #1 这个命令调用 可能是在子任务中 可能是整个任务 --不改
        self.subtask = self._start_subtask
        #2 不在子任务中 那就重置为最开始--不改 index这些设置完了
        if self.subtask not in self.SUBTASK_RESET_INDEX.keys():
            obs = self.env.reset() 
            self._elapsed_steps = 0 
        #2 如果是子任务调用的  复杂一点 重置为最开始 然后示教动作做完前面的动作  然后到目标子任务的开始状态
        else:
            success = False
            while not success:
                #1 不成功 重置 不改
                obs = self.env.reset() 
                self.subtask = self._start_subtask
                self._elapsed_steps = 0 
                #拿动作不改
                action, skill_index = self.env.get_oracle_action(obs)
                count, max_steps = 0, self.SUBTASK_RESET_MAX_STEPS[self.subtask]
                #2 不断拿施教动作去step  直到subtak的index发生变化 或者总步长达到-step不改 之前检查了
                while skill_index < self.SUBTASK_RESET_INDEX[self.subtask] and count < max_steps:
                    obs, reward, done, info = self.env.step(action)
                    action, skill_index = self.env.get_oracle_action(obs)
                    count += 1

                #3 奇怪的逻辑  观察和信息流程需要梳理一遍  --未完成
                with self.switch_subtask():
                    obs_ = self._replace_goal_with_subgoal(obs.copy())  # in case repeatedly replace goal
                    success = self.compute_reward(obs_['achieved_goal'], obs_['desired_goal']) + 1

        if self._output_raw_obs: return self._replace_goal_with_subgoal(obs), obs
        else: return self._replace_goal_with_subgoal(obs)
    #TODO 未完成
    def _replace_goal_with_subgoal(self, obs):
        """Replace ag and g"""
        subgoal = self._subgoal()    
        kukacol = pairwise_collision(self.env.obj_id, self._kuka.body)

        if self.subtask == 'grasp':
            obs['achieved_goal'] = np.concatenate([obs['observation'][0: 3], obs['observation'][7: 10], [kukacol]])

        elif self.subtask == 'release':
            obs['achieved_goal'] = np.concatenate([obs['observation'][7: 10], obs['achieved_goal'], [kukacol]])
        obs['desired_goal'] = np.append(subgoal, self.SUBTASK_CONTACT_CONDITION[self.subtask])
        return obs
    #不用改
    def _subgoal(self):
        """Output goal of subtask"""
        goal = self.env.subgoals[self.SUBTASK_ORDER[self.subtask]]
        return goal
    #TODO 未完成
    def compute_reward(self, ag, g, info=None):
        """Compute reward that indicates the success of subtask"""
        if len(ag.shape) == 1:
            if self.subtask == 'release':
                goal_reach = self.env.compute_reward(ag[-5:-2], g[-5:-2], None) + 1
            else:
                goal_reach = self.env.compute_reward(ag[:-2], g[:-2], None) + 1
            contact_cond = np.all(ag[-2:]==g[-2:])
            reward = (goal_reach and contact_cond) - 1
        else:
            if self.subtask == 'release':
                goal_reach = self.env.compute_reward(ag[:,-5:-2], g[:,-5:-2], None).reshape(-1, 1) + 1
            else:
                goal_reach = self.env.compute_reward(ag[:,:-2], g[:,:-2], None).reshape(-1, 1) + 1
            contact_cond = np.all(ag[:, -2:]==g[:, -2:], axis=1).reshape(-1, 1)
            reward = np.all(np.hstack([goal_reach, contact_cond]), axis=1) - 1.
        return reward


class KukagraspSCWrapper(KukagraspSLWrapper):
    '''Wrapper for skill chaining.'''
    MAX_ACTION_RANGE = 4.
    REWARD_SCALE = 30.
    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        next_obs_ = self._replace_goal_with_subgoal(next_obs.copy())
        reward = self.compute_reward(next_obs_['achieved_goal'], next_obs_['desired_goal'])
        info['step'] = 1 - reward
        done = self._elapsed_steps == self.SUBTASK_STEPS[self.subtask]
        reward = done * reward 
        info['subtask'] = self.subtask
        info['subtask_done'] = False
        info['subtask_is_success'] = reward 

        if done:
            info['subtask_done'] = True
            # Transit to next subtask (if current subtask is not terminal) and reset elapsed steps
            if self.subtask in self.SUBTASK_NEXT_SUBTASK.keys():
                done = False
                self._elapsed_steps = 0
                self.subtask = self.SUBTASK_NEXT_SUBTASK[self.subtask]
                info['is_success'] = False
                reward = 0
            else:
                info['is_success'] = reward 
            next_obs_ = self._replace_goal_with_subgoal(next_obs)

        if self._output_raw_obs: return next_obs_, reward, done, info, next_obs
        else: return next_obs_, reward, done, info

    def reset(self, subtask=None):
        self.subtask = self._start_subtask if subtask is None else subtask
        if self.subtask not in self.SUBTASK_RESET_INDEX.keys():
            obs = self.env.reset() 
            self._elapsed_steps = 0 
        else:
            success = False
            while not success:
                obs = self.env.reset() 
                self.subtask = self._start_subtask if subtask is None else subtask
                self._elapsed_steps = 0 

                action, skill_index = self.env.get_oracle_action(obs)
                count, max_steps = 0, self.SUBTASK_RESET_MAX_STEPS[self.subtask]
                while skill_index < self.SUBTASK_RESET_INDEX[self.subtask] and count < max_steps:
                    obs, reward, done, info = self.env.step(action)
                    action, skill_index = self.env.get_oracle_action(obs)
                    count += 1

                # Reset again if failed
                with self.switch_subtask():
                    obs_ = self._replace_goal_with_subgoal(obs.copy())  # in case repeatedly replace goal
                    success = self.compute_reward(obs_['achieved_goal'], obs_['desired_goal']) + 1

        if self._output_raw_obs: return self._replace_goal_with_subgoal(obs), obs
        else: return self._replace_goal_with_subgoal(obs)

    #---------------------------Reward---------------------------
    def compute_reward(self, ag, g, info=None):
        """Compute reward that indicates the success of subtask"""
        if len(ag.shape) == 1:
            if self.subtask == 'release':
                goal_reach = self.env.compute_reward(ag[-5:-2], g[-5:-2], None) + 1
            else:
                goal_reach = self.env.compute_reward(ag[:-2], g[:-2], None) + 1
            contact_cond = np.all(ag[-2:]==g[-2:])
            reward = (goal_reach and contact_cond) - 1
        else:
            if self.subtask == 'release':
                goal_reach = self.env.compute_reward(ag[:,-5:-2], g[:,-5:-2], None).reshape(-1, 1) + 1
            else:
                goal_reach = self.env.compute_reward(ag[:,:-2], g[:,:-2], None).reshape(-1, 1) + 1
            if self.subtask == 'grasp':
                raise NotImplementedError
            contact_cond = np.all(ag[:, -2:]==g[:, -2:], axis=1).reshape(-1, 1)
            reward = np.all(np.hstack([goal_reach, contact_cond]), axis=1) - 1.
        return reward + 1

    def goal_adapator(self, goal, subtask, device=None):
        '''Make predicted goal compatible with wrapper'''
        if isinstance(goal, np.ndarray):
            return np.append(goal, self.SUBTASK_CONTACT_CONDITION[subtask])
        elif isinstance(goal, torch.Tensor):
            assert device is not None
            ct_cond = torch.tensor(self.SUBTASK_CONTACT_CONDITION[subtask], dtype=torch.float32)
            ct_cond = ct_cond.repeat(goal.shape[0], 1).to(device)
            adp_goal = torch.cat([goal, ct_cond], 1)
            return adp_goal

    def get_reward_functions(self):
        reward_funcs = {}
        for subtask in self.subtask_order.keys():
            with self.switch_subtask(subtask):
                reward_funcs[subtask] = self.compute_reward
        return reward_funcs

    @property
    def start_subtask(self):
        return self._start_subtask

    @property
    def max_episode_steps(self):
        assert np.sum([x for x in self.SUBTASK_STEPS.values()]) == self.env._max_episode_steps
        return self.env._max_episode_steps

    @property
    def max_action_range(self):
        return self.MAX_ACTION_RANGE

    @property
    def subtask_order(self):
        return self.SUBTASK_ORDER
    
    @property
    def subtask_steps(self):
        return self.SUBTASK_STEPS
    
    @property
    def subtasks(self):
        subtasks = []
        for subtask, order in self.subtask_order.items():
            if order >= self.subtask_order[self.start_subtask]:
                subtasks.append(subtask)
        return subtasks

    @property
    def prev_subtasks(self):
        return self.SUBTASK_PREV_SUBTASK 
    
    @property
    def next_subtasks(self):
        return self.SUBTASK_NEXT_SUBTASK 
    
    @property
    def last_subtask(self):
        return self.LAST_SUBTASK

    @property
    def len_cond(self):
        return len(self.SUBTASK_CONTACT_CONDITION[self.last_subtask])
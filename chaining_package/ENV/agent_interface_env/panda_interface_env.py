import abc
from contextlib import contextmanager

import gym
import numpy as np
import torch


def pairwise_collision(bullet_client,body1, body2, max_distance=0):  # 10000
    # getContactPoints
    # return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance)) != 0

    return bullet_client.getContactPoints(body1, body2) != ()


from memory_profiler import profile

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
        #env step的时候走这里
        if subtask is not None:
            curr_subtask = self.subtask            
            self.subtask = subtask
            yield
            self.subtask = curr_subtask
        #env reset的时候走这里
        else:
            #看起来是切换为上一个任务 可是为什么？
            self.subtask = self.SUBTASK_PREV_SUBTASK[self.subtask]
            yield
            self.subtask = self.SUBTASK_NEXT_SUBTASK[self.subtask]


#-----------------------------Kukagrasp-v0-v0-----------------------------
class PandaGraspSLWrapper(SkillLearningWrapper):
    '''Wrapper for skill learning'''
    #子任务顺序--完成
    SUBTASK_ORDER = {
        'grasp': 0,
        'move':1,
        'release':2
    }    
    #子任务步长env step的步长吧--完成
    SUBTASK_STEPS = {
        'grasp': 20,
        'move':35,
        'release':5
    }
    #需要reset到的子任务的开始路点是多少？
    SUBTASK_RESET_INDEX = {
        'move':3,
        'release': 5
    }
    
    #重置到一个子任务开始状态 环境首先需要执行多少步--完成
    SUBTASK_RESET_MAX_STEPS = {
        'move':20,
        'release': 55
    }
    #对之前任务的了解-完成
    SUBTASK_PREV_SUBTASK = {
        'move':'grasp',
        'release': 'move'
    }
    #对下一个任务的了解--完成
    SUBTASK_NEXT_SUBTASK = {
        'grasp': 'move',
        'move':'release'
    }
    #子任务的接触条件--完成--
    #根据源代码  认为是在这个动作中 要不要接触物体的一种指引  如果你机械臂在这个动作中 需要最终抓到物体 那就是1 不要 那就是0
    SUBTASK_CONTACT_CONDITION = {
        'grasp': [1],
        'move':[1],
        'release': [ 0]
    }
    #最后一个子任务--完成
    LAST_SUBTASK = 'release'
    def __init__(self, env, subtask='release', output_raw_obs=False):
        super().__init__(env, subtask, output_raw_obs)
        self.done_subtasks = {key: False for key in self.SUBTASK_STEPS.keys()}
        self.find_reward=False
        if subtask =='grasp':
            self.env.env.grasp_or_release=True
        elif subtask =='move':
            self.env.env.grasp_or_release=False    
        elif subtask =='release':
            self.env.env.grasp_or_release=False
        #检查点1 初始化效果
        #breakpoint() #2 就是设置了一些属性 也看不出对不对

    @property
    def max_episode_steps(self):
        # breakpoint() #这里出过错 调整的权宜之计是register的时候写死
        assert np.sum([x for x in self.SUBTASK_STEPS.values()]) == self.env._max_episode_steps #根据调试器的信息 这东西果然是register的时候写的那个东西

        return self.SUBTASK_STEPS[self.subtask]

    def step(self, action):
        #print("搜集数据")
        next_obs, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        next_obs_ = self._replace_goal_with_subgoal(next_obs.copy())
        # if self.env.env.state_save_id ==1:
        # print(f"到达的目标{next_obs_['achieved_goal']}  \n 期望到达的目标{next_obs_['desired_goal']}")

        reward = self.compute_reward(next_obs_['achieved_goal'], next_obs_['desired_goal'])
        # print(f"此时的奖励{reward}")
        info['is_success'] = reward + 1
        done = self._elapsed_steps == self.max_episode_steps
        # save groud truth goal
        #这个东西到底是干什么的？
        with self.switch_subtask(self.LAST_SUBTASK):
            info['gt_goal'] = self._replace_goal_with_subgoal(next_obs.copy())['desired_goal']
  
        if self._output_raw_obs: return next_obs_, reward, done, info, next_obs
        else: return next_obs_, reward, done, info,
    
    # @profile(stream=open('接口环境的env reset泄漏问题.txt','w'))

    def reset(self):
        #1 这个命令调用 可能是在子任务中 可能是整个任务 --不改
        self.subtask = self._start_subtask

        #2 第一个任务走这里
        if self.subtask not in self.SUBTASK_RESET_INDEX.keys():
            obs = self.env.reset() 
            self._elapsed_steps = 0 
        #2 后续的任务走这里 --这里的目的是切换到子任务的开始状态   
        else: 
            success = False
            while not success:
                #检查点2 重置效果
                #breakpoint()
                obs = self.env.reset() 
                self.subtask = self._start_subtask
                self._elapsed_steps = 0 
                #拿动作不改
                #检查点3 示教动作
                #breakpoint()
                action, skill_index = self.env.get_oracle_action(obs)

                count, max_steps = 0, self.SUBTASK_RESET_MAX_STEPS[self.subtask]
                #2 不断拿施教动作去step  直到subtak的index发生变化 或者总步长达到-step不改 之前检查了
                #检查点4 检查是示教运行到某个子任务之前
                
                while skill_index < self.SUBTASK_RESET_INDEX[self.subtask] and count < max_steps:
                    obs, reward, done, info = self.env.step(action)
                    action, skill_index = self.env.get_oracle_action(obs)
                    count += 1

                # print(f"执行完reset的step之后 skill_index是多少  {skill_index}")

                #到这里的时候 可以认为是 执行了 前面的所有的子任务的施教技能  
                #3 在这里需要知道什么？ 需要知道究竟前面的施教执行成功没有  真正的核心不是replace 而是 comutereward  
                    #之前的那个子任务的goal信息 需要用之前子任务写在这个类中的信息来拿到  但是是临时的 只是为了判断是否成功  所以使用context manager
                
                #检查点5 检查任务临时切换和奖励计算结果


                with self.switch_subtask():
                    obs_ = self._replace_goal_with_subgoal(obs.copy())  # in case repeatedly replace goal
                    success = self.compute_reward(obs_['achieved_goal'], obs_['desired_goal']) + 1
                #     print(f"执行完reset的step之后 到达的目标 {obs_['achieved_goal']}")
                #     print(f"执行完reset的step之后 期望的目标 {obs_['desired_goal']}")
                # print(f"reset是否成功?  {success}")
                # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        if self._output_raw_obs: return self._replace_goal_with_subgoal(obs), obs
        else: return self._replace_goal_with_subgoal(obs)

    def _replace_goal_with_subgoal(self, obs):
        """Replace ag and g"""
        #检查点6 检查子目标拿的对不对
        #breakpoint() 设函数里面 别设外面
        subgoal = self._subgoal()  
        # print(f"subgoal {subgoal}") 
        #添加奇怪的接触信息  
        #检查点7 检查传入参数id是否正确  接触信息是否获取到  
        #breakpoint() # 第一次检查完毕 正常

        contact = pairwise_collision(self.env.env._p,self.env.blockUid, self.env.pandauid) #所谓的body 源代码拿得是什么东西？psm1的body  这东西怎么来的？loadurdf文件返回的   机器人在pybullet中的uid sdf
        
        # print(f"有没有撞倒？{kukacol}")
        #在不同的子任务的时候  调整到达目标 接触信息加上 不一样的位置信息
        #抓取的时候达到的目的  是两个机械臂的位置
        if contact == 1 :
            self.find_reward =True
        else :
            self.find_reward =False
        # if self.env.env.state_save_id ==1:
        #     breakpoint()
        # if self.subtask == 'grasp':
        #     obs['achieved_goal'] = np.concatenate([obs['achieved_goal'],[kukacol,]])
        # if self.subtask == 'move':
        #     obs['achieved_goal'] = np.concatenate([obs['achieved_goal'],[kukacol,]])

        # #释放的时候大概率是物体位置反正无所谓了  不行再改了      
        # elif self.subtask == 'release':
        #     obs['achieved_goal'] = np.concatenate([obs['achieved_goal'],[kukacol,]])
        # if self.subtask=='move':
        #     breakpoint()
        obs['achieved_goal'] = np.concatenate([obs['achieved_goal'],[contact,]])
        obs['desired_goal'] = np.append(subgoal, self.SUBTASK_CONTACT_CONDITION[self.subtask])
        # print(f"现在的任务 {self.subtask}  期望的目标:{obs['desired_goal']} 达到的目标{obs['achieved_goal']}")
        
        # obs['desired_goal'] =np.round(obs['desired_goal'] ,4)
        #这两个goal需要同样的维度吗？从源代码看 并不是
        #检查点8  检查传入参数id是否正确  接触信息是否获取到
        #breakpoint() #并未检查出问题 两个goal 一个是7维度 一个四维度 看不出区别
        #print(f"replace里面的东西{obs['achieved_goal'] }")
        return obs
    #不用改
    def _subgoal(self):
        """Output goal of subtask"""
        goal = self.env.subgoals[self.SUBTASK_ORDER[self.subtask]]
        #breakpoint()# 4 有两个子目标 分别是抓取上来和释放的位置 都是物体的 看起来没问题
        return goal
    #TODO 未完成
    def compute_reward(self, ag, g, info=None):
        """Compute reward that indicates the success of subtask"""
        #检查点8 检查传入的信息 更改奖励计算的逻辑
        #breakpoint()
        #更改了goal之后 使用原始的env的奖励计算是不可能的


        if len(ag.shape) == 1:
            #还是需要原来的奖励函数的

            if self.subtask != 'release':
                goal_reach = self.env.compute_reward(ag[:3], g[:3], None) + 1
            else :

                goal_reach = self.env.compute_reward(ag[:3], g[:3], None) + 1
                #print(f"到达的位置：{ag[:3]} \n 期望到达的位置{g[:3]}")
 
            # 在原来的奖励函数的基础上加碰撞条件的奖励 
            contact_cond = np.all(ag[-1:]==g[-1:])#注意切片的问题
            #print(f"看看是不是切片的问题{ag[-2:]}")
            reward = (goal_reach and contact_cond) - 1
            
            # if contact_cond:
            #     reward = (goal_reach ) - 1 +0.5
            # else :
            #     reward = (goal_reach ) - 1
            # if self.find_reward:
            #     with open("找末端抖动没奖励原因.txt","a") as file:
            #         file.write(f"此时的goalreach奖励{goal_reach}\n")   
            #         file.write(f"此时的contact_cond奖励{contact_cond}\n")   
            #         file.write(f"此时的总奖励{reward}\n")  



            #TODO 出bug了 施教动作到不了位置
        else:
            if self.subtask == 'release':
                goal_reach = self.env.compute_reward(ag[:,:-1], g[:,:-1], None).reshape(-1, 1) + 1
            else:
                goal_reach = self.env.compute_reward(ag[:,:-1], g[:,:-1], None).reshape(-1, 1) + 1
            contact_cond = np.all(ag[:, -1:]==g[:, -1:], axis=1).reshape(-1, 1)
            reward = np.all(np.hstack([goal_reach, contact_cond]), axis=1) - 1.
        
        return reward


class PandaGraspSCWrapper(PandaGraspSLWrapper):
    '''Wrapper for skill chaining.'''
    MAX_ACTION_RANGE = 4.
    REWARD_SCALE = 30.
    def step(self, action):
        #保证底下拿上来的obs里面的achieved goal是对的 
        if self.subtask =='grasp':
            self.env.env.grasp_or_release=True
        elif self.subtask =='move':
            self.env.env.grasp_or_release=False    
        elif self.subtask =='release':
            self.env.env.grasp_or_release=False
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


            goal_reach = self.env.compute_reward(ag[:3], g[:3], None) + 1



            contact_cond = np.all(ag[-1:]==g[-1:])
            reward = (goal_reach and contact_cond) - 1
        else:
            #batch的数据
            goal_reach = self.env.compute_reward(ag[:,:3], g[:,:3], None).reshape(-1, 1) + 1
            contact_cond = np.all(ag[:,-1:]==g[:,-1:], axis=1).reshape(-1, 1)
            reward = np.all(np.hstack([goal_reach, contact_cond]), axis=1) - 1.

        
        # if len(ag.shape) == 1:
        #     if self.subtask == 'release':
        #         goal_reach = self.env.compute_reward(ag[-5:-2], g[-5:-2], None) + 1
        #     else:
        #         goal_reach = self.env.compute_reward(ag[:-2], g[:-2], None) + 1
        #     contact_cond = np.all(ag[-2:]==g[-2:])
        #     reward = (goal_reach and contact_cond) - 1
        # else:
        #     if self.subtask == 'release':
        #         goal_reach = self.env.compute_reward(ag[:,-5:-2], g[:,-5:-2], None).reshape(-1, 1) + 1
        #     else:
        #         goal_reach = self.env.compute_reward(ag[:,:-2], g[:,:-2], None).reshape(-1, 1) + 1
        #     if self.subtask == 'grasp':
        #         raise NotImplementedError
        #     contact_cond = np.all(ag[:, -2:]==g[:, -2:], axis=1).reshape(-1, 1)
        #     reward = np.all(np.hstack([goal_reach, contact_cond]), axis=1) - 1.
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
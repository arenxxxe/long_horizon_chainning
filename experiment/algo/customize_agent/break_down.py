# from ..viskill_agents.utils.general_utils import AttrDict, listdict2dictlist
# from ..viskill_agents.utils.rl_utils import ReplayCache, ReplayCacheGT
import hydra

import torch
import torch.nn as nn
import torch.nn.functional as F
##依赖有哪些？
##
from ..viskill_agents.factory import make_sl_agent
from ..viskill_agents.modules.replay_buffer import HER_sampler_seq


from gym import spaces
import numpy as np


import os

##模仿真正的输入来生成东西
class MimicEnv:
     def __init__(self,cfg):
        #取到cfg
          self.cfg=cfg
        #agent里面需要的env参数 写死
          #接口分析：env提供 一个类似字典的东西  动作空间 动作上限  动作随机采样器  
        #2024/5/10 
               #模仿env里面设置动作和观察空间
          self.env_params={}
          ACTION_SIZE = 5  # (dx, dy, dz, dyaw/dpitch, open/close)#根据机械臂自由度和电机来设置这个东西
          self.action_size=ACTION_SIZE #
          self.action_space = spaces.Box(-1, 1, shape=(self.action_size,), dtype='float32')
          
          self.env_params['act']=self.action_space.shape[0]
          self.env_params['max_action']= self.action_space.high[0]
          self.env_params['act_rand_sampler'] = self.action_space.sample
               #模仿env里面设置观察空间
          #需要模拟的观察
          self.goal=np.array([0,1,2]) #物体的目标位置
          achieved_goal=np.array([0,0,0]) #一般是机械臂末端执行器位置  原始代码中 拿着物体时候是路点位置或者物体位置
          robot_state=np.ones(7) #机械臂末端执行器位置  7个自由度 末端执行的三位置 三欧拉角 最后一个估计是关节自由度 
          #物体的位置
          pos=[1,2,3]
          object_pos = np.array(pos)
          object_rel_pos = object_pos - robot_state[0: 3]
          #waypoint 不知道真正的含义 只能硬猜是和物体链接的虚拟路径点  为了保证物体更加平滑的运动
          waypoint_pos = np.array([1, 2, 3])
          waypoint_rot = np.array([0, 0, 0])#欧拉角 也是三个
          observation = np.concatenate([
            robot_state, object_pos.ravel(), object_rel_pos.ravel(),
            waypoint_pos.ravel(), waypoint_rot.ravel()  # achieved_goal.copy(),
        ])
          obs = {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy()
        }
          self.observation_space = spaces.Dict(dict(
          desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
          achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
          observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
          ))
          self.env_params['obs']=self.observation_space['observation'].shape[0]
          self.env_params['goal']=self.observation_space['desired_goal'].shape[0]
     
     def get_env_params(self):
          return self.env_params
     def compute_reward(self, ag, g, info=None):
          #真正计算奖励的地方 一个一个子任务计算
          #就是比较到达的目标和最终目标（子任务目标 被wrapper替换了）
          if len(ag.shape) == 1:
               #为什么这些goal和真正env中的goal不同？
               #因为这些goal是env wrapper引入的
     #                   if self.subtask == 'grasp':
     #        obs['achieved_goal'] = np.concatenate([obs['observation'][0: 3], obs['observation'][7: 10], [psm1col, psm2col]])#后面这两个是机械臂和物体的碰撞
     #    elif self.subtask == 'handover':
     #        obs['achieved_goal'] = np.concatenate([obs['observation'][7: 10], obs['observation'][0: 3], [psm1col, psm2col]])
     #    elif self.subtask == 'release':
     #        obs['achieved_goal'] = np.concatenate([obs['observation'][7: 10], obs['achieved_goal'], [psm1col, psm2col]])
     #    obs['desired_goal'] = np.append(subgoal, self.SUBTASK_CONTACT_CONDITION[self.subtask])
               if self.subtask == 'release':
                    #self._is_success(achieved_goal, desired_goal).astype(np.float32) - 1.
                    #        d = goal_distance(achieved_goal, desired_goal)
                    #     return (d < self.distance_threshold).astype(np.float32)
                    ##计算输入的达到目标和子目标/最终目标之间的距离小于一个阈值 来计算是否成功
                    #大于阈值 这东西回来0 不成功回来-1 加一之后正好是true false整数值
                    goal_reach = self.env.compute_reward(ag[-5:-2], g[-5:-2], None) + 1
               else:
                    goal_reach = self.env.compute_reward(ag[:-2], g[:-2], None) + 1
               contact_cond = np.all(ag[-2:]==g[-2:])
               #到达目标 而且接触条件成立 那么啥也不给 要是不行 直接惩罚
               reward = (goal_reach and contact_cond) - 1
          else:
               if self.subtask == 'release':
                    goal_reach = self.env.compute_reward(ag[:,-5:-2], g[:,-5:-2], None).reshape(-1, 1) + 1
               else:
                    goal_reach = self.env.compute_reward(ag[:,:-2], g[:,:-2], None).reshape(-1, 1) + 1
               contact_cond = np.all(ag[:, -2:]==g[:, -2:], axis=1).reshape(-1, 1)
               reward = np.all(np.hstack([goal_reach, contact_cond]), axis=1) - 1.
          return reward



# #用于测试agent基本功能的sampler  无需env 随机生成数据
# class SimpleTest:
   
#     def __init__(self, env, max_episode_len):

#         self._env=env
        
#         self._max_episode_len = max_episode_len

#         self._obs = None
#         self._episode_step = 0
#         self._episode_cache = ReplayCacheGT(max_episode_len)
#     #测试agent的正常获得 和参数读入过程
#     def test_make_agent(self, num_episodes, render=False):
#          self.agent=make_sl_agent()
#          pass
#     #测试env正常的输入输出
#     def test_env(self, num_episodes, render=False):
#          pass
#     #测试replaybuffer的效果
#     def test_replay_buffer(self, num_episodes, render=False):
#          pass
#     #测试agent的保存和读取
#     def test_agent_save_load(self, num_episodes, render=False):
#          pass
#     #测试本地logger和wandb的正常运作：
#     def test_logger(self, num_episodes, render=False):
#          pass
#     #测试demo buffer 数据完整性 数据格式 数据量  数据分类的逻辑
#     def test_demo_buffer(self, num_episodes, render=False):
#          pass

def load_npz():

     demo_path = os.path.join(os.getcwd(),'experiment\\algo\customize_agent\data')
     file_name = "data_test"
     file_name += ".npz"

     demo_path = os.path.join(demo_path, file_name)

     arr1 = np.array([1, 2, 3])
     arr2 = np.array([[4, 5], [6, 7]])

# 将数组写入npz文件
     np.savez( demo_path, arr1=arr1, arr2=arr2)


     demo = np.load(demo_path,allow_pickle=True)

     return demo


@hydra.main(version_base=None, config_path="../../configs", config_name="skill_learning")
def main(cfg):
    #保证关键的接口  拿到环境  拿到采样器 采样器能采出需要的格式的数据  拿这些数据训练
     

     env=MimicEnv(cfg)
     env_params = env.get_env_params()
     #拿采样器
     buffer_sampler = HER_sampler_seq(
     replay_strategy=cfg.agent.sampler.strategy,
     replay_k=cfg.agent.sampler.k,
     reward_func=env.compute_reward,
     )
     # make 出agent 
     agent = make_sl_agent(env_params, buffer_sampler, cfg.agent)
     data=load_npz()
     import pdb;pdb.set_trace()

     SimpleTest = SimpleTest(env,agent, 100)

     episode, rollouts, env_steps = train_sampler.sample_episode(is_train=True, render=False)
     
     #################################不急着训练 先保证输入输出的东西是看起来正常的东西#######################################
     # # 环境搜集这个过程的状态和时间记录
     # rollout_storage.append(episode)
     # rollout_status = rollout_storage.rollout_stats()
     # self._global_step += int(mpi_sum(env_steps))#所有的worker的步数
     # self._global_episode += int(mpi_sum(1))

     # # 轮次数据存进buffer
     # self.buffer.store_episode(rollouts)
     # self.agent.update_normalizer(rollouts)          

     # # 反向传播
     # if not seed_until_steps(ep_start_step):
          
     #     metrics = self.agent.update(self.buffer, self.demo_buffer)








if __name__ == '__main__':
    main()
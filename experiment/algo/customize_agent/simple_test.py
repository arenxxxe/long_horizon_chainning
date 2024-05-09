# from ..viskill_agents.utils.general_utils import AttrDict, listdict2dictlist
# from ..viskill_agents.utils.rl_utils import ReplayCache, ReplayCacheGT
import hydra

import torch
import torch.nn as nn
import torch.nn.functional as F
# from ..viskill_agents.factory import make_sl_agent


##模仿真正的输入来生成东西
class MimicEnv:
    def __init__(self,cfg):
        self.init_obs=None
        self.cfg=cfg




#用于测试agent基本功能的sampler  无需env 随机生成数据
class SimpleTest:
   
    def __init__(self, env, max_episode_len):

        self._env=env
        
        self._max_episode_len = max_episode_len

        self._obs = None
        self._episode_step = 0
        self._episode_cache = ReplayCacheGT(max_episode_len)
    #测试agent的正常获得 和参数读入过程
    def test_make_agent(self, num_episodes, render=False):
         self.agent=make_sl_agent()
         pass
    #测试env正常的输入输出
    def test_env(self, num_episodes, render=False):
         pass
    #测试replaybuffer的效果
    def test_replay_buffer(self, num_episodes, render=False):
         pass
    #测试agent的保存和读取
    def test_agent_save_load(self, num_episodes, render=False):
         pass
    #测试本地logger和wandb的正常运作：
    def test_logger(self, num_episodes, render=False):
         pass
    #测试demo buffer 数据完整性 数据格式 数据量  数据分类的逻辑
    def test_demo_buffer(self, num_episodes, render=False):
         pass




@hydra.main(version_base=None, config_path="../../configs", config_name="skill_learning")
def main(cfg):
    #保证关键的接口  拿到环境  拿到采样器 采样器能采出需要的格式的数据  拿这些数据训练
        import pdb;pdb.set_trace()
        env=MimicEnv(cfg)

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
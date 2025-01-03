import os
import torch

from chaining_package.ALGO.viskill_agent.algo.viskill_agents.factory import make_sl_agent
from chaining_package.ALGO.viskill_agent.algo.viskill_agents.components.checkpointer import CheckpointHandler, save_cmd
from chaining_package.ALGO.viskill_agent.algo.viskill_agents.components.envrionment import make_env
from chaining_package.ALGO.viskill_agent.algo.viskill_agents.components.logger import Logger, WandBLogger, logger
from chaining_package.ALGO.viskill_agent.algo.viskill_agents.modules.replay_buffer import HerReplayBufferWithGT, get_buffer_sampler
from chaining_package.ALGO.viskill_agent.algo.viskill_agents.modules.sampler import Sampler
from chaining_package.ALGO.viskill_agent.algo.viskill_agents.utils.general_utils import (AttrDict,AverageMeter, Every, Timer, Until,
                                   set_seed_everywhere)
# from chaining_package.ALGO.viskill_agent.algo.viskill_agents.utils.mpi import (mpi_gather_experience_episode,
#                          mpi_gather_experience_rollots, mpi_sum,
#                          update_mpi_config)
from chaining_package.ALGO.viskill_agent.algo.viskill_agents.utils.rl_utils import RolloutStorage, get_env_params, init_demo_buffer
from chaining_package.EXPERIMENT.training.trainer.my_trainer.trainers.base_trainer import BaseTrainer
from chaining_package.ENV.agent_interface_env.kuka_slsc_wrapper import KukagraspSLWrapper

#检查内存泄漏问题

from pympler import asizeof
import psutil
#文件输出自定义
memory_record =0
eval_true=False

# def wirte_to_profle_txt(filename):
#     def decorater(func):
#         def wrapper(*args,**kwargs):
#             f= open(filename,'a')
#             try:
#                 @profile(stream=f)
#                 def wrapped(*args,**kwargs):
#                     global memory_record

#                     available_memory=psutil.virtual_memory().available / (1024**2)
#                     f.write(f'系统当前可用内存{available_memory}MB\n')
#                     f.write(f"和刚才执行之后的内存变化{available_memory-memory_record}MB\n")
#                     memory_record=available_memory
#                     return func(*args,**kwargs)
#                 return wrapped(*args,**kwargs)
#             finally:
#                 f.close()
        
#         return wrapper
#     return decorater


class EVAL(BaseTrainer):
    def _setup(self):
        self._setup_env()       # Environment
        self._setup_buffer()    # Relay buffer
        self._setup_agent()     # Agent
        self._setup_sampler()   # Sampler
        self._setup_logger()    # Logger
        self._setup_misc()      # MISC
        if self.is_chef:
            self.termlog.info('Setup done')

    def _setup_env(self):
        #一个env配置一个client

        self.train_env = make_env(self.cfg,state_save_id=1)
        self.eval_env = make_env(self.cfg,state_save_id=0)
        self.env_params = get_env_params(self.train_env, self.cfg)


        
    def _setup_buffer(self):
        self.buffer_sampler = get_buffer_sampler(self.train_env, self.cfg.agent.sampler)
        self.buffer = HerReplayBufferWithGT(buffer_size=self.cfg.replay_buffer_capacity, env_params=self.env_params,
                            batch_size=self.cfg.batch_size, sampler=self.buffer_sampler)
        self.demo_buffer = HerReplayBufferWithGT(buffer_size=self.cfg.replay_buffer_capacity, env_params=self.env_params,
                            batch_size=self.cfg.batch_size, sampler=self.buffer_sampler)
        
    def _setup_agent(self):
        self.agent = make_sl_agent(self.env_params, self.buffer_sampler, self.cfg.agent)

    def _setup_sampler(self):
        self.train_sampler = Sampler(self.train_env, self.agent, self.env_params['max_timesteps'])
        self.eval_sampler = Sampler(self.eval_env, self.agent, self.env_params['max_timesteps'])

    def _setup_logger(self):
        # update_mpi_config(self.cfg)
        if self.is_chef:
            exp_name = f"EVAL_1_{self.cfg.task}_{self.cfg.subtask}__{self.cfg.agent.name}__seed{self.cfg.seed}"
            if self.cfg.postfix is not None:
                exp_name =  exp_name + '__' + self.cfg.postfix 
            self.wb = WandBLogger(exp_name=exp_name, project_name=self.cfg.project_name, entity=self.cfg.entity_name, \
                    path=self.work_dir, conf=self.cfg)
            self.logger = Logger(self.work_dir)
            self.termlog = logger
            save_cmd(self.work_dir)
        else: 
            self.wb, self.logger, self.termlog = None, None, None
    
    def _setup_misc(self):

        init_demo_buffer(self.cfg, self.demo_buffer, self.agent)

        if self.is_chef:
            self.model_dir = self.work_dir / 'model'
            self.model_dir.mkdir(exist_ok=True)
            #不清理原本的checkpoint
            # for file in os.listdir(self.model_dir):
            #     os.remove(self.model_dir / file)

        self.device = torch.device(self.cfg.device)
        self.timer = Timer()
        self._global_step = 0
        self._global_episode = 0
        set_seed_everywhere(self.cfg.seed)
        #没有chekpoint直接注释掉
        self.best_checkpoint_episode=0
        self.load_checkpoint()

    def load_checkpoint(self):
        self.best_checkpoint_episode=CheckpointHandler.load_checkpoint(self.model_dir,self.agent,self.device,episode="best")
    

    def train(self):

        n_train_episodes = int(self.cfg.n_train_steps / self.env_params['max_timesteps'])
        n_eval_episodes = int(n_train_episodes / self.cfg.n_eval) * 1
        n_save_episodes = int(n_train_episodes / self.cfg.n_save) * 1
        n_log_episodes = int(n_train_episodes / self.cfg.n_log) * 1
        
        assert n_save_episodes > n_eval_episodes
        if n_save_episodes % n_eval_episodes != 0:
            n_save_episodes = int(n_save_episodes / n_eval_episodes) * n_eval_episodes

        train_until_episode = Until(n_train_episodes)
        save_every_episodes = Every(n_save_episodes)
        eval_every_episodes = Every(n_eval_episodes)

        log_every_episodes = Every(n_log_episodes)
        seed_until_steps = Until(self.cfg.n_seed_steps)
        if self.is_chef:
            self.termlog.info('Starting training')
        
        
        while train_until_episode(self.global_episode):





            # self._train_episode(log_every_episodes, seed_until_steps)




            #print(self.global_episode)
            # if eval_every_episodes(self.global_episode):
            print("开始测试")
            score = self.eval()

            if not self.cfg.dont_save and save_every_episodes(self.global_episode) and self.is_chef:
                filename =  CheckpointHandler.get_ckpt_name(self.global_episode+self.best_checkpoint_episode)
                # TODO(tao): expose scoring metric
                CheckpointHandler.save_checkpoint({
                    'episode': self.global_episode+self.best_checkpoint_episode,
                    'global_step': self.global_step,
                    'state_dict': self.agent.state_dict(),
                    'o_norm': self.agent.o_norm,
                    'g_norm': self.agent.g_norm,
                    'score': score,
                }, self.model_dir, filename)
                if self.cfg.save_buffer: 
                    self.buffer.save(self.model_dir, self.global_episode)
                self.termlog.info(f'Save checkpoint to {os.path.join(self.model_dir, filename)}')
    
 
    def _train_episode(self, log_every_episodes, seed_until_steps):
        # import tracemalloc
        # tracemalloc.start()

        # snapshot1=tracemalloc.take_snapshot()

        # sync network parameters across workers
        # if self.use_multiple_workers:
        #     self.agent.sync_networks()

        self.timer.reset()
        batch_time = AverageMeter()
        ep_start_step = self.global_step
        metrics = None

        # collect experience
        rollout_storage = RolloutStorage()
        ########################################################## 主要增长源之一
        episode, rollouts, env_steps = self.train_sampler.sample_episode(is_train=True, render=False)
        #################################################################

        # if self.use_multiple_workers:
        #     rollouts = mpi_gather_experience_episode(rollouts)

        # update status
            
        ########################################没有任何增长 不能认为是  
        rollout_storage.append(episode)


        ########################################################## 不认为下面这句的泄漏有问题 因为这东西增很少
        rollout_status = rollout_storage.rollout_stats()

        # with open("训练期奖励记录.txt","a") as file:
        #         file.write(f"第{self.global_episode}rollout记录完成 总的奖励是{rollout_status.avg_reward}\n")

        self._global_step += int(env_steps)
        self._global_episode += int(1)

        # save to buffer
        self.buffer.store_episode(rollouts)
        # self.agent.update_normalizer(rollouts)          

        # update policy
        # if not seed_until_steps(ep_start_step):
        #     if self.is_chef:
        # ########################################################## 不认为是主要增长源头
        #         metrics = self.agent.update(self.buffer, self.demo_buffer)
       

        #     if self.use_multiple_workers:
        # ##########下面不可能泄漏 条件一直不成立
                
        #         self.agent.sync_networks()
                

        # snapshot2=tracemalloc.take_snapshot()

        # top_stat=snapshot2.compare_to(snapshot1,'lineno')
        # with open('leak_inspect.txt','a') as f:
        #     for i,stat in enumerate(top_stat[:10],1):
        #         f.write('-'*40+'\n')
        #         f.write(f'#{i}:{stat}')
        #         for line in stat.traceback.format():
        #             f.write(line+'\n')
        #         f.write(f'memory block: {stat.count}\n')
        #         f.write(f"new allocate Size : {stat.size/1024:.2f} KB \n")
        #         f.write('-'*40+'\n')
        #print(f"全局步数{self.global_episode}")
        # if metrics is not None :
        #     print(f"计算出的metircs{metrics}")
        # if self.is_chef :
        #     print("是主线程")
        # if log_every_episodes(self.global_episode) :
        #     print("到了该log的时候了")
        #     breakpoint()
        # log results
        if metrics is not None and log_every_episodes(self.global_episode) and self.is_chef:
            elapsed_time, total_time = self.timer.reset()
            batch_time.update(elapsed_time)
            togo_train_time = batch_time.avg * (self.cfg.n_train_steps - ep_start_step) / env_steps / 1

            self.logger.log_metrics(metrics, self.global_step, ty='train')
            with self.logger.log_and_dump_ctx(self.global_step, ty='train') as log:
                print("进入普通的terminal log")
                log('fps', env_steps / elapsed_time)
                log('total_time', total_time)
                log('episode_reward', rollout_status.avg_reward)
                log('episode_length', env_steps)
                log('episode_sr', rollout_status.avg_success_rate)
                log('episode', self.global_episode)
                log('step', self.global_step)
                log('ETA', togo_train_time)
            #print(f"看看是否正常在计算东西{metrics}")
            self.wb.log_outputs(metrics, None, log_images=False, step=self.global_step, is_train=True)

    def eval(self):
        '''Eval agent.'''

        eval_rollout_storage = RolloutStorage()
        for _ in range(self.cfg.n_eval_episodes):
            eval_true=True
            # if eval_true:
            #     #print("刚进测试")

            # episode, _, env_steps = self.eval_sampler.sample_episode(is_train=False, render=True,eval_true=eval_true)
            episode, _, env_steps = self.eval_sampler.sample_episode(is_train=False, render=True)

            eval_rollout_storage.append(episode)
        # if eval_true:
        #     print("出了采样器了")

        eval_true=False
        rollout_status = eval_rollout_storage.rollout_stats()

        from functools import reduce
        import numpy as np
        def listdict2dictlist(LD):
            """ Converts a list of dicts to a dict of lists """
            
            # Take intersection of keys
            keys = reduce(lambda x,y: x & y, (map(lambda d: d.keys(), LD)))
            return AttrDict({k: [dic[k] for dic in LD] for k in keys})

        def joinListDict(LD):
            """Joins a list of dictionaries that contain lists."""
            DL = listdict2dictlist(LD)
            return type(LD[0])({k: np.array(DL[k]) for k in DL})
        def no_mpi_gather_experience_rollots(experience_rollouts):
            buf=[experience_rollouts]
            return joinListDict(buf)
        

        # if self.use_multiple_workers:
        if True:
            rollout_status = no_mpi_gather_experience_rollots(rollout_status)
            for key, value in rollout_status.items():
                rollout_status[key] = value.mean()

        if self.is_chef:
            self.wb.log_outputs(rollout_status, eval_rollout_storage, log_images=True, step=self.global_step)
            with self.logger.log_and_dump_ctx(self.global_step, ty='eval') as log:
                log('episode_sr', rollout_status.avg_success_rate)
                log('episode_reward', rollout_status.avg_reward)
                log('episode_length', env_steps)
                log('episode', self.global_episode)
                log('step', self.global_step)

        del eval_rollout_storage
        return rollout_status.avg_success_rate 

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def is_chef(self):
        return True

    @property
    def use_multiple_workers(self):
        return True
    

import hydra    
@hydra.main(version_base=None, config_path="../my_trainer/configs", config_name="skill_learning")
def main(cfg):

    exp = EVAL(cfg)
    exp.train()

if __name__ == "__main__":

    main()
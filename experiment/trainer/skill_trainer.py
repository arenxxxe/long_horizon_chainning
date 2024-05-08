from base import BaseTrainer



class SkillLearningTrainer(BaseTrainer):
    def _setup(self):
        self._setup_env()       # Environment
        self._setup_buffer()    # Relay buffer
        self._setup_agent()     # Agent
        self._setup_sampler()   # Sampler
        self._setup_logger()    # Logger
        self._setup_misc()      # MISC


        #terminal提示设置完成
        self.termlog.info('Setup done')

    def _setup_env(self):
        #拿两个环境 一个是训练的环境 一个是测试的环境
        self.train_env = make_env(self.cfg)
        self.eval_env = make_env(self.cfg)
        #用cfg 的参数配置好
        self.env_params = get_env_params(self.train_env, self.cfg)
        
    def _setup_buffer(self):
        #三个东西 一个从buffer中采样的东西 两个buffer 一个训练搜集到的 一个示教数据
        self.buffer_sampler = get_buffer_sampler(self.train_env, self.cfg.agent.sampler)
        self.buffer = HerReplayBufferWithGT(buffer_size=self.cfg.replay_buffer_capacity, env_params=self.env_params,
                            batch_size=self.cfg.batch_size, sampler=self.buffer_sampler)
        self.demo_buffer = HerReplayBufferWithGT(buffer_size=self.cfg.replay_buffer_capacity, env_params=self.env_params,
                            batch_size=self.cfg.batch_size, sampler=self.buffer_sampler)
        
    def _setup_agent(self):
        #搞出神经网络对象
        self.agent = make_sl_agent(self.env_params, self.buffer_sampler, self.cfg.agent)

    def _setup_sampler(self):
        #拿着agent和buffer 和环境交互的采样器 也就是搜集数据 存到buffer里面 
        #训练一个 测试一个
        self.train_sampler = Sampler(self.train_env, self.agent, self.env_params['max_timesteps'])
        self.eval_sampler = Sampler(self.eval_env, self.agent, self.env_params['max_timesteps'])

    def _setup_logger(self):
        #日志记录
        #设定好每次日志的名字 执行任务 子任务 agent名字 种子名字 后缀
        exp_name = f"SL_{self.cfg.task}_{self.cfg.subtask}_{self.cfg.agent.name}_seed{self.cfg.seed}"
        if self.cfg.postfix is not None:
        exp_name =  exp_name + '_' + self.cfg.postfix 
        #一个普通的本地日志 一个wanb上传到网页可以看的日志
        self.wb = WandBLogger(exp_name=exp_name, project_name=self.cfg.project_name, entity=self.cfg.entity_name, \
                path=self.work_dir, conf=self.cfg)
        self.logger = Logger(self.work_dir)
        self.termlog = logger
        #把执行的命令记下来
        save_cmd(self.work_dir)

    
    def _setup_misc(self):
        #demo的数据装进来
        init_demo_buffer(self.cfg, self.demo_buffer, self.agent)

        #建一个model文件夹 保存训练中的checkpoint模型 如果有了就清空model文件夹中的文件
        self.model_dir = self.work_dir / 'model'
        self.model_dir.mkdir(exist_ok=True)
        for file in os.listdir(self.model_dir):
                os.remove(self.model_dir / file)
        #神经网络需要指定设备 时间和步骤的记录初始化
        self.device = torch.device(self.cfg.device)
        self.timer = Timer()
        self._global_step = 0
        self._global_episode = 0
        #全局的设置训练种子
        set_seed_everywhere(self.cfg.seed)
    
    def train(self):

        n_train_episodes = int(self.cfg.n_train_steps / self.env_params['max_timesteps'])
        n_eval_episodes = int(n_train_episodes / self.cfg.n_eval) * self.cfg.mpi.num_workers
        n_save_episodes = int(n_train_episodes / self.cfg.n_save) * self.cfg.mpi.num_workers
        n_log_episodes = int(n_train_episodes / self.cfg.n_log) * self.cfg.mpi.num_workers
        # 确保存储周期是n_eval_episodes的倍数
        assert n_save_episodes > n_eval_episodes
        if n_save_episodes % n_eval_episodes != 0:
            n_save_episodes = int(n_save_episodes / n_eval_episodes) * n_eval_episodes
        #指示是否结束
        train_until_episode = Until(n_train_episodes)
        save_every_episodes = Every(n_save_episodes)
        eval_every_episodes = Every(n_eval_episodes)
        log_every_episodes = Every(n_log_episodes)
        seed_until_steps = Until(self.cfg.n_seed_steps)

        #提示开始训练
        self.termlog.info('Starting training')
        #训练
        while train_until_episode(self.global_episode):
            #一个轮次的训练
            self._train_episode(log_every_episodes, seed_until_steps)
                
            # 如果需要 一个轮次评估一次效果
            if eval_every_episodes(self.global_episode):
                score = self.eval()
                #保存模型和轮次训练输出导出为zip压缩文件
            if not self.cfg.dont_save and save_every_episodes(self.global_episode):
                filename =  CheckpointHandler.get_ckpt_name(self.global_episode)
                # TODO(tao): expose scoring metric
                CheckpointHandler.save_checkpoint({
                    'episode': self.global_episode,
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
        # 多worker搜集的网络参数同步
        #isaac sim里面都batch 的数据传进传出 只用一个神经网络 大概率不需要

        self.timer.reset()
        batch_time = AverageMeter()
        ep_start_step = self.global_step
        metrics = None

        # 轮次数据搜集 仿真环境交互
        rollout_storage = RolloutStorage()
        episode, rollouts, env_steps = self.train_sampler.sample_episode(is_train=True, render=False)


        # 环境搜集这个过程的状态和时间记录
        rollout_storage.append(episode)
        rollout_status = rollout_storage.rollout_stats()
        self._global_step += int(mpi_sum(env_steps))#所有的worker的步数
        self._global_episode += int(mpi_sum(1))

        # 轮次数据存进buffer
        self.buffer.store_episode(rollouts)
        self.agent.update_normalizer(rollouts)          

        # 反向传播
        if not seed_until_steps(ep_start_step):
            if self.is_chef:
                metrics = self.agent.update(self.buffer, self.demo_buffer)
            if self.use_multiple_workers:
                self.agent.sync_networks()

        # 日志记录
        if metrics is not None and log_every_episodes(self.global_episode) and self.is_chef:
            elapsed_time, total_time = self.timer.reset()
            batch_time.update(elapsed_time)
            togo_train_time = batch_time.avg * (self.cfg.n_train_steps - ep_start_step) / env_steps / self.cfg.mpi.num_workers

            self.logger.log_metrics(metrics, self.global_step, ty='train')
            with self.logger.log_and_dump_ctx(self.global_step, ty='train') as log:
                log('fps', env_steps / elapsed_time)
                log('total_time', total_time)
                log('episode_reward', rollout_status.avg_reward)
                log('episode_length', env_steps)
                log('episode_sr', rollout_status.avg_success_rate)
                log('episode', self.global_episode)
                log('step', self.global_step)
                log('ETA', togo_train_time)
            self.wb.log_outputs(metrics, None, log_images=False, step=self.global_step, is_train=True)
            
    def eval(self):
        '''Eval agent.'''
        #数据存储
        eval_rollout_storage = RolloutStorage()
        for _ in range(self.cfg.n_eval_episodes):
            episode, _, env_steps = self.eval_sampler.sample_episode(is_train=False, render=True)
            eval_rollout_storage.append(episode)
        rollout_status = eval_rollout_storage.rollout_stats()
        #多神经网络时候的数据存储
        if self.use_multiple_workers:
            rollout_status = mpi_gather_experience_rollots(rollout_status)
            for key, value in rollout_status.items():
                rollout_status[key] = value.mean()

        #数据日志记录 wandb上传 普通logger展示到terminal然后记录dump出来
        self.wb.log_outputs(rollout_status, eval_rollout_storage, log_images=True, step=self.global_step)
        with self.logger.log_and_dump_ctx(self.global_step, ty='eval') as log:
                log('episode_sr', rollout_status.avg_success_rate)
                log('episode_reward', rollout_status.avg_reward)
                log('episode_length', env_steps)
                log('episode', self.global_episode)
                log('step', self.global_step)
        #释放内存 草真的假的？ 这是动态语言？
        del eval_rollout_storage
        return rollout_status.avg_success_rate 

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode



    @property
    def use_multiple_workers(self):
        #从issac的env里面调出来吧 不使用多线程
        return 
    



############辅助的东西 按照执行顺序排##################


######初始化并配置环境
    
def make_env(cfg)：
    env = gym.make(cfg.task)
    if cfg.task == 'L-v0':
        if cfg.skill_chaining:
            env = LongHSCWrapper(env, cfg.init_subtask, output_raw_obs=False)
        else:
            env = LongHSLWrapper(env, cfg.subtask, output_raw_obs=False)
    else:
        raise NotImplementedError
    return env


class LongHSLWrapper(SkillLearningWrapper):
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
        'handover': 45,
        'release': 70
    }
    SUBTASK_PREV_SUBTASK = {
        'handover': 'grasp',
        'release': 'handover'
    }
    SUBTASK_NEXT_SUBTASK = {
        'grasp': 'handover',
        'handover': 'release'
    }
    SUBTASK_CONTACT_CONDITION = {
        'grasp': [0, 1],
        'handover': [1, 0],
        'release': [0, 0]
    }
    LAST_SUBTASK = 'release'
    def __init__(self, env, subtask='grasp', output_raw_obs=False):
        super().__init__(env, subtask, output_raw_obs)
        self.done_subtasks = {key: False for key in self.SUBTASK_STEPS.keys()}

    @property
    def max_episode_steps(self):
        assert np.sum([x for x in self.SUBTASK_STEPS.values()]) == self.env._max_episode_steps
        return self.SUBTASK_STEPS[self.subtask]

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
                    obs_ = self._replace_goal_with_subgoal(obs.copy())  # in case repeatedly replace goal
                    success = self.compute_reward(obs_['achieved_goal'], obs_['desired_goal']) + 1

        if self._output_raw_obs: return self._replace_goal_with_subgoal(obs), obs
        else: return self._replace_goal_with_subgoal(obs)

    def _replace_goal_with_subgoal(self, obs):
        """Replace ag and g"""
        subgoal = self._subgoal()    
        psm1col = pairwise_collision(self.env.obj_id, self.psm1.body)
        psm2col = pairwise_collision(self.env.obj_id, self.psm2.body)

        if self.subtask == 'grasp':
            obs['achieved_goal'] = np.concatenate([obs['observation'][0: 3], obs['observation'][7: 10], [psm1col, psm2col]])
        elif self.subtask == 'handover':
            obs['achieved_goal'] = np.concatenate([obs['observation'][7: 10], obs['observation'][0: 3], [psm1col, psm2col]])
        elif self.subtask == 'release':
            obs['achieved_goal'] = np.concatenate([obs['observation'][7: 10], obs['achieved_goal'], [psm1col, psm2col]])
        obs['desired_goal'] = np.append(subgoal, self.SUBTASK_CONTACT_CONDITION[self.subtask])
        return obs

    def _subgoal(self):
        """Output goal of subtask"""
        goal = self.env.subgoals[self.SUBTASK_ORDER[self.subtask]]
        return goal

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



class SkillLearningWrapper(gym.Wrapper):
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



class LongHSCWrapper(LongHSLWrapper):
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


#env的向量维度 动作维度 最大时间步等信息 对神经网络构建相关的吧
def get_env_params(train_env, cfg):
    obs = train_env.reset()
    env_params = AttrDict(
        obs=obs['observation'].shape[0],
        achieved_goal=obs['achieved_goal'].shape[0],
        goal=obs['desired_goal'].shape[0],
        act=train_env.action_space.shape[0],
        act_rand_sampler=train_env.action_space.sample,
        max_timesteps=train_env.max_episode_steps,
        max_action=train_env.action_space.high[0],
    )
    if cfg.skill_chaining:
        env_params.update(AttrDict(
            act_sc=obs['achieved_goal'].shape[0] - train_env.len_cond, # withoug contact condition
            max_action_sc=train_env.max_action_range,
            adaptor_sc=train_env.goal_adapator,
            subtask_order=train_env.subtask_order,
            num_subtasks=len(train_env.subtask_order),
            subtask_steps=train_env.subtask_steps,
            subtasks=train_env.subtasks,
            next_subtasks=train_env.next_subtasks,
            prev_subtasks=train_env.prev_subtasks,
            middle_subtasks=train_env.next_subtasks.keys(),
            last_subtask=train_env.last_subtask,
            reward_funcs=train_env.get_reward_functions(),
            len_cond=train_env.len_cond
        ))
    return env_params


######初始化并配置神经网络
#经验采样 只有训练用到
def get_buffer_sampler(train_env, sampler_cfg):
    if sampler_cfg.type == 'her':
        sampler = HER_sampler(
            replay_strategy=sampler_cfg.strategy,
            replay_k=sampler_cfg.k,
            reward_func=train_env.compute_reward,
        )
    elif sampler_cfg.type == 'her_seq':
        sampler = HER_sampler_seq(
            replay_strategy=sampler_cfg.strategy,
            replay_k=sampler_cfg.k,
            reward_func=train_env.compute_reward,
        )
    else:
        raise NotImplementedError
    return sampler



class HER_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        
        # to get the params to re-compute reward
        transitions['r'] = self.reward_func(transitions['ag_next'], transitions['g'], None)
        if len(transitions['r'].shape) == 1:
            transitions['r'] = np.expand_dims(transitions['r'], 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions


class HER_sampler_seq(HER_sampler):
    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T-1, size=batch_size)   # from T to T-1
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}

        next_actions = episode_batch['actions'][episode_idxs, t_samples + 1].copy()
        transitions['next_actions'] = next_actions
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward
        transitions['r'] = self.reward_func(transitions['ag_next'], transitions['g'], None)
        if len(transitions['r'].shape) == 1:
            transitions['r'] = np.expand_dims(transitions['r'], 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions
    


class HerReplayBufferWithGT:

    def __init__(self, env_params, buffer_size, batch_size, sampler, T=None):
        # TODO(tao): unwrap env_params
        self.env_params = env_params
        self.T = T if T is not None else env_params['max_timesteps']
        self.size = buffer_size // self.T
        self.batch_size = batch_size
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sampler
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['achieved_goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['act']]),
                        'dones': np.empty([self.size, self.T, 1]),
                        'gt_g': np.empty([self.size, self.T, self.env_params['goal']]),
                        }
        self.sample_keys = ['obs', 'ag', 'g', 'actions', 'dones']
    
    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions, dones, mb_gt_g = episode_batch.obs, episode_batch.ag, episode_batch.g, \
                                                    episode_batch.actions, episode_batch.dones, episode_batch.gt_g
        batch_size = mb_obs.shape[0]
        idxs = self._get_storage_idx(inc=batch_size)

        # store the informations
        self.buffers['obs'][idxs] = mb_obs
        self.buffers['ag'][idxs] = mb_ag
        self.buffers['g'][idxs] = mb_g
        self.buffers['actions'][idxs] = mb_actions
        self.buffers['dones'][idxs] = dones
        self.buffers['gt_g'][idxs] = mb_gt_g
        self.n_transitions_stored += self.T * batch_size
    
    # sample the data from the replay buffer
    def sample(self):
        temp_buffers = {}
        for key in self.sample_keys:
            temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func.sample_her_transitions(temp_buffers, self.batch_size)
        return transitions

#神经网络
#经典工厂模式 
AGENTS = {
    'SL_DDPGBC': SkillLearningDDPGBC,
    'SL_DEX': SkillLearningDEX,
    'SC_DDPG': SkillChainingDDPG,
    'SC_AWAC': SkillChainingAWAC,
    'SC_SAC': SkillChainingSAC,
    'SC_SAC_SIL': SkillChainingSACSIL,
}



def make_sl_agent(env_params, buffer_sampler, agent_cfg):
    if agent_cfg.name not in AGENTS.keys():
        assert 'Agent is not supported: %s' % agent_cfg.name
    else:
        assert 'SL' in agent_cfg.name 
        return AGENTS[agent_cfg.name](
            env_params=env_params,
            sampler=buffer_sampler,
            agent_cfg=agent_cfg
        )


#环境交互
class Sampler:

    
    """Collects rollouts from the environment using the given agent."""
    def __init__(self, env, agent, max_episode_len):
        self._env = env
        self._agent = agent
        self._max_episode_len = max_episode_len

        self._obs = None
        self._episode_step = 0
        self._episode_cache = ReplayCacheGT(max_episode_len)

    def init(self):
        """Starts a new rollout. Render indicates whether output should contain image."""
        self._episode_reset()

    def sample_action(self, obs, is_train):
        return self._agent.get_action(obs, noise=is_train)

    def sample_episode(self, is_train, render=False):
        """Samples one episode from the environment."""
        self.init()
        episode, done = [], False
        while not done and self._episode_step < self._max_episode_len:
            action = self.sample_action(self._obs, is_train)
            if action is None:
                break
            if render:
                render_obs = self._env.render('rgb_array')
            obs, reward, done, info = self._env.step(action)
            episode.append(AttrDict(
                reward=reward,
                success=info['is_success'],
                info=info
            ))
            self._episode_cache.store_transition(obs, action, done, info['gt_goal'])
            if render:
                episode[-1].update(AttrDict(image=render_obs))

            # update stored observation
            self._obs = obs
            self._episode_step += 1

        episode[-1].done = True     # make sure episode is marked as done at final time step
        rollouts = self._episode_cache.pop()
        assert self._episode_step == self._max_episode_len
        return listdict2dictlist(episode), rollouts, self._episode_step

    def _episode_reset(self, global_step=None):
        """Resets sampler at the end of an episode."""
        self._episode_step, self._episode_reward = 0, 0.
        self._obs = self._reset_env()
        self._episode_cache.store_obs(self._obs)

    def _reset_env(self):
        return self._env.reset()


#日志
class   WandBLogger:
    """Logs to WandB."""
    N_LOGGED_SAMPLES = 50    # how many examples should be logged in each logging step

    def __init__(self, exp_name, project_name, entity, path, conf, exclude=None):
        """
        :param exp_name: full name of experiment in WandB
        :param project_name: name of overall project
        :param entity: name of head entity in WandB that hosts the project
        :param path: path to which WandB log-files will be written
        :param conf: hyperparam config that will get logged to WandB
        :param exclude: (optional) list of (flattened) hyperparam names that should not get logged
        """
        if exclude is None: exclude = []
        flat_config = flatten_dict(conf)
        filtered_config = {k: v for k, v in flat_config.items() if (k not in exclude and not inspect.isclass(v))}
        
        # clear dir
        # save_dir = path / 'wandb'
        # save_dir.mkdir(exist_ok=True)
        # shutil.rmtree(f"{save_dir}/")

        logger.info("Init wandb")
        wandb.init(
            resume=exp_name,
            project=project_name,
            config=filtered_config,
            dir=path,
            entity=entity,
            notes=conf.notes if 'notes' in conf else ''
        )

    def log_scalar_dict(self, d, prefix='', step=None):
        """Logs all entries from a dict of scalars. Optionally can prefix all keys in dict before logging."""
        if prefix: d = prefix_dict(d, prefix + '_')
        wandb.log(d) if step is None else wandb.log(d, step=step)

    def log_videos(self, vids, name, step=None):
        """Logs videos to WandB in mp4 format.
        Assumes list of numpy arrays as input with [time, channels, height, width]."""
        assert len(vids[0].shape) == 4 and vids[0].shape[1] == 3
        assert isinstance(vids[0], np.ndarray)
        if vids[0].max() <= 1.0: vids = [np.asarray(vid * 255.0, dtype=np.uint8) for vid in vids]
        # TODO(karl) expose the FPS as a parameter
        log_dict = {name: [wandb.Video(vid, fps=10, format="mp4") for vid in vids]}
        wandb.log(log_dict) if step is None else wandb.log(log_dict, step=step)

    def log_plot(self, fig, name, step=None):
        """Logs matplotlib graph to WandB.
        fig is a matplotlib figure handle."""
        img = wandb.Image(fig)
        wandb.log({name: img}) if step is None else wandb.log({name: img}, step=step)

    def log_outputs(self, logging_stats, rollout_storage, log_images, step, is_train=False, log_videos=True, log_video_caption=False):
        """Visualizes/logs all training outputs."""
        self.log_scalar_dict(logging_stats, prefix='train' if is_train else 'eval', step=step)

        if log_images:
            assert rollout_storage is not None      # need rollout data for image logging
            # log rollout videos with info captions
            if 'image' in rollout_storage and log_videos:
                if log_video_caption:
                    vids = [np.stack(add_captions_to_seq(rollout.image, np2obj(rollout.info))).transpose(0, 3, 1, 2)
                            for rollout in rollout_storage.get()[-self.n_logged_samples:]]
                else:
                    vids = [np.stack(rollout.image).transpose(0, 3, 1, 2)
                            for rollout in rollout_storage.get()[-self.n_logged_samples:]]
                self.log_videos(vids, name="rollouts", step=step)

    @property
    def n_logged_samples(self):
        # TODO(karl) put this functionality in a base logger class + give it default parameters and config
        return self.N_LOGGED_SAMPLES
   
class    Logger:
    def __init__(self, log_dir):
        self._log_dir = log_dir
        self._train_mg = MetersGroup(log_dir / 'train.csv',
                                     formating=COMMON_TRAIN_FORMAT)
        self._eval_mg = MetersGroup(log_dir / 'eval.csv',
                                    formating=COMMON_EVAL_FORMAT)
        self._sw = None

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def log(self, key, value, step):
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value, step)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value)

    def log_metrics(self, metrics, step, ty):
        for key, value in metrics.items():
            self.log(f'{ty}/{key}', value, step)

    def dump(self, step, ty=None):
        if ty is None or ty == 'eval':
            self._eval_mg.dump(step, 'eval')
        if ty is None or ty == 'train':
            self._train_mg.dump(step, 'train')

    def log_and_dump_ctx(self, step, ty):
        return LogAndDumpCtx(self, step, ty)

def save_cmd(base_dir):
  train_cmd = 'python ' + ' '.join([sys.argv[0]] + [pipes.quote(s) for s in sys.argv[1:]])
  train_cmd += '\n\n'
  print('\n' + '*' * 80)
  print('Training command:\n' + train_cmd)
  print('*' * 80 + '\n')
  with open(os.path.join(base_dir, "cmd.txt"), "a") as f:
    f.write(train_cmd)

def init_demo_buffer(cfg, buffer, agent, subtask=None, update_normalizer=True):
    '''Load demonstrations into buffer and initilaize normalizer'''
    demo_path = os.path.join(os.getcwd(),'surrol/data/demo')
    file_name = "data_"
    file_name += cfg.task
    file_name += "_" + 'random'
    if subtask is None:
        file_name += "_" + str(cfg.num_demo) + '_primitive_new' + cfg.subtask
    else:
        file_name += "_" + str(cfg.num_demo) + '_primitive_new' + subtask
    file_name += ".npz"

    demo_path = os.path.join(demo_path, file_name)
    demo = np.load(demo_path, allow_pickle=True)
    demo_obs, demo_acs, demo_gt = demo['observations'], demo['actions'], demo['gt_actions']

    episode_cache = ReplayCacheGT(buffer.T)
    for epsd in range(cfg.num_demo):
        episode_cache.store_obs(demo_obs[epsd][0])
        for i in range(buffer.T):
            episode_cache.store_transition(
                obs=demo_obs[epsd][i+1],
                action=demo_acs[epsd][i],
                done=i==(buffer.T-1),
                gt_goal=demo_gt[epsd][i]
            )
        episode = episode_cache.pop()
        buffer.store_episode(episode)
        if update_normalizer:
            agent.update_normalizer(episode)




##############train过程
#判断是否到达设定时间步
class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False
    
class CheckpointHandler:
    @staticmethod
    def get_ckpt_name(episode):
        return 'weights_ep{}.pth'.format(episode)

    @staticmethod
    def get_episode(path):
        checkpoint_names = glob.glob(os.path.abspath(path) + "/*.pth")
        if len(checkpoint_names) == 0:
            logger.error("No checkpoints found at {}!".format(path))
        processed_names = [file.split('/')[-1].replace('weights_ep', '').replace('.pth', '')
                           for file in checkpoint_names]
        episodes = list(filter(lambda x: x is not None, [str2int(name) for name in processed_names]))
        return episodes
    
    @staticmethod
    def get_resume_ckpt_file(resume, path):
        episodes = CheckpointHandler.get_episode(path)
        file_paths = [os.path.join(path, CheckpointHandler.get_ckpt_name(episode)) for episode in episodes]
        scores = [torch.load(file_path)['score'] for file_path in file_paths]
        if resume == 'latest':
            max_episode = np.max(episodes)
            resume_file = CheckpointHandler.get_ckpt_name(max_episode)
            logger.info(f'Checkpoints with max episode {max_episode} with the success rate {scores[np.argmax(episodes)]}!')
        elif resume == 'best':
            max_episode = episodes[get_last_argmax(scores)]
            resume_file = CheckpointHandler.get_ckpt_name(max_episode)
            logger.info(f'Checkpoints with success rate {scores}, the highest success rate {max(scores)}!')
        return os.path.join(path, resume_file), max_episode

    @staticmethod
    def save_checkpoint(state, folder, filename='checkpoint.pth'):
        torch.save(state, os.path.join(folder, filename))
        
    @staticmethod
    def load_checkpoint(checkpt_dir, agent, device, episode='best'):
        """Loads weigths from checkpoint."""
        checkpt_path, max_episode = CheckpointHandler.get_resume_ckpt_file(episode, checkpt_dir)
        checkpt = torch.load(checkpt_path, map_location=device)
    
        logger.info(f'Loading pre-trained model from {checkpt_path}!')
        agent.load_state_dict(checkpt['state_dict'])
        if 'g_norm' in checkpt.keys() and 'o_norm' in checkpt.keys():
            agent.g_norm = checkpt['g_norm']
            agent.o_norm = checkpt['o_norm']


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, digits=None):
        """
        :param digits: number of digits returned for average value
        """
        self._digits = digits
        self.reset()

    def reset(self):
        self.val = 0
        self._avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self._avg = self.sum / max(1, self.count)

    @property
    def avg(self):
        if self._digits is not None:
            return np.round(self._avg, self._digits)
        else:
            return self._avg

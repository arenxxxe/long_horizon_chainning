import numpy as np

#环境交互
class Tester:

    
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

class ReplayCache:
    def __init__(self, T):
        self.T = T
        self.reset()

    def reset(self):
        self.t = 0
        self.obs, self.ag, self.g, self.actions, self.dones = [], [], [], [], []

    def store_transition(self, obs, action, done):
        self.obs.append(obs['observation'])
        self.ag.append(obs['achieved_goal'])
        self.g.append(obs['desired_goal'])
        self.actions.append(action)
        self.dones.append(done)

    def store_obs(self, obs):
        self.obs.append(obs['observation'])
        self.ag.append(obs['achieved_goal'])

    def pop(self):
        assert len(self.obs) == self.T + 1 and len(self.actions) == self.T
        obs = np.expand_dims(np.array(self.obs.copy()),axis=0)
        ag = np.expand_dims(np.array(self.ag.copy()), axis=0)
        #print(self.ag)
        g = np.expand_dims(np.array(self.g.copy()), axis=0)
        actions = np.expand_dims(np.array(self.actions.copy()), axis=0)
        dones = np.expand_dims(np.array(self.dones.copy()), axis=1)
        dones = np.expand_dims(dones, axis=0)

        self.reset()
        episode = AttrDict(obs=obs, ag=ag, g=g, actions=actions, dones=dones)
        return episode


class ReplayCacheGT(ReplayCache):
    def reset(self):
        self.t = 0
        self.obs, self.ag, self.g, self.actions, self.dones, self.gt_g = [], [], [], [], [], []

    def store_transition(self, obs, action, done, gt_goal):
        self.obs.append(obs['observation'])
        self.ag.append(obs['achieved_goal'])
        self.g.append(obs['desired_goal'])
        self.actions.append(action)
        self.dones.append(done)
        self.gt_g.append(gt_goal)

    def pop(self):
        assert len(self.obs) == self.T + 1 and len(self.actions) == self.T
        obs = np.expand_dims(np.array(self.obs.copy()),axis=0)
        ag = np.expand_dims(np.array(self.ag.copy()), axis=0)
        #print(self.ag)
        g = np.expand_dims(np.array(self.g.copy()), axis=0)
        actions = np.expand_dims(np.array(self.actions.copy()), axis=0)
        dones = np.expand_dims(np.array(self.dones.copy()), axis=1)
        dones = np.expand_dims(dones, axis=0)
        gt_g = np.expand_dims(np.array(self.gt_g.copy()), axis=0)

        self.reset()
        episode = AttrDict(obs=obs, ag=ag, g=g, actions=actions, dones=dones, gt_g=gt_g)
        return episode
        

def main():
    #保证关键的接口  拿到环境  拿到采样器 采样器能采出需要的格式的数据  拿这些数据训练
        train_sampler = Sampler(self.train_env, self.agent, self.env_params['max_timesteps'])
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



###########helper
    
class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d    









if __name__ == '__main__':
    main()
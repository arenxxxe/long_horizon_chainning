import time
import socket

import gym
from gym import spaces
from gym.utils import seeding
from pybullet_utils import bullet_client as bc
import pybullet

import pybullet_data
import pkgutil

def step(bullet_client,duration=1.0):
    for i in range(int(duration * 240)):
        bullet_client.stepSimulation()
def render_image(bullet_client,width, height, view_matrix, proj_matrix, shadow=1):
    (_, _, px, _, mask) = bullet_client.getCameraImage(width=width,
                                           height=height,
                                           viewMatrix=view_matrix,
                                           projectionMatrix=proj_matrix,
                                           shadow=shadow,
                                           lightDirection=(10, 0, 10),
                                           renderer=bullet_client.ER_BULLET_HARDWARE_OPENGL)

    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (height, width, 4))

    rgb_array = rgb_array[:, :, :3]
    return rgb_array, mask


import numpy as np
from memory_profiler import profile

RENDER_HEIGHT = 480  # train
RENDER_WIDTH = 640
# RENDER_HEIGHT = 1080  # record
# RENDER_WIDTH = 1920


class SurRoLEnv(gym.Env):
    """
    A gym Env wrapper for SurRoL.
    refer to: https://github.com/openai/gym/blob/master/gym/core.py
    """

    metadata = {'render.modes': ['human', 'rgb_array', 'img_array']}

    def __init__(self, render_mode: str = None):
        # rendering and connection options
        self._render_mode = render_mode
        # render_mode = 'human'
        # if render_mode == 'human':
        #     self.cid = p.connect(p.SHARED_MEMORY)
        #     if self.cid < 0:
        if render_mode == 'human':
            self._p = bc.BulletClient(connection_mode=pybullet.GUI)

            # self._p.connect(self._p.GUI)
            # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self._p = bc.BulletClient(connection_mode=pybullet.DIRECT)

            # See PyBullet Quickstart Guide Synthetic Camera Rendering
            # TODO: no light when using direct without egl
            
            # if socket.gethostname().startswith('pc') or True:
            #     # TODO: not able to run on remote server
            #     egl = pkgutil.get_loader('eglRenderer')
                
            #     plugin = self._p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        # camera related setting
        self._view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=(-0.05 * self.SCALING-0.3, 0, 0.36 * self.SCALING),
                                                                distance=2,
                                                                yaw=90,
                                                                pitch=-5.0,
                                                                roll=0,
                                                                upAxisIndex=2)
        self._proj_matrix = self._p.computeProjectionMatrixFOV(fov=45,
                                                         aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                         nearVal=0.1,
                                                         farVal=20.0)
        # additional settings
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.configureDebugVisualizer(lightPosition=(10.0, 0.0, 10.0))
        self._p.setGravity(0, 0, -9.81)
        #地板位置调整 0没有问题
        self._p.loadURDF("plane.urdf", (0, 0, 0))
        #获取物体信息的开始
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}

        self.seed()

        # self.actions = []  # only for demo
        
        self._env_setup()
        
        step(self._p,0.25)
        

        self.goal = self._sample_goal()  # tasks are all implicitly goal-based

        self._sample_goal_callback()


               
        obs = self._get_obs()

       
        self.action_space = spaces.Box(-1, 1, shape=(self.action_size,), dtype='float32')
        if isinstance(obs, np.ndarray):
            # gym.Env
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')
        elif isinstance(obs, dict):
            # gym.GoalEnv
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
            ))
        else:
            raise NotImplementedError
        self._duration = 0.2  # important for mini-steps

    def step(self, action: np.ndarray):
        # action should have a shape of (action_size, )
        if len(action.shape) > 1:
            action = action.squeeze(axis=-1)
        action = np.clip(action, self.action_space.low*2, self.action_space.high*2)
        # import wandb
        # wandb.log({"x":action[0],
        #     "y":action[1],
        #     "z":action[2]
            
        #     })
        # time0 = time.time()
        self._set_action(action)
        # time1 = time.time()
        # TODO: check the best way tao step simulation
        #step(self._duration)

        # time2 = time.time()
        # print(" -> robot action time: {:.6f}, simulation time: {:.4f}".format(time1 - time0, time2 - time1))
        self._step_callback()
        obs = self._get_obs()

        done = False

        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        } if isinstance(obs, dict) else {'achieved_goal': None}
        if isinstance(obs, dict):
            reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        else:
            reward = self.compute_reward(obs, self.goal, info)
        # if len(self.actions) > 0:
        #     self.actions[-1] = np.append(self.actions[-1], [reward])  # only for demo
        return obs, reward, done, info
    # @profile(stream=open('surrol goal env的底层reset泄漏问题.txt','w'))
    def reset(self):
        # reset scene in the corresponding file

        #必须这么做 每次分配的id是变化的 不清空 一直递增把内存爆了不说 拿过时的id调pybullet接口肯定出错
        
        #不这么做 怀疑这里造成了泄漏
        # self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}

        # p.resetSimulation()
        # p.setGravity(0, 0, -9.81)

        self._p.restoreState(fileName=f"{self.state_save_id}_state.bullet")

        self._p.configureDebugVisualizer(lightPosition=(10.0, 0.0, 10.0))

        # Temporarily disable rendering to load scene faster.
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
        # p.configureDebugVisualizer(p.COV_ENABLE_PLANAR_REFLECTION, 0)

        # p.loadURDF("plane.urdf", (0, 0, 0))
        #这里就得多承担一点 
        # reset机械臂
        # reset两个物体
        # reset那个红色路点

        # self._env_setup()
        #step(0.25)
        self.goal = self._sample_goal().copy()
        self._sample_goal_callback()

        # Re-enable rendering.
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

        obs = self._get_obs()
        return obs

    def close(self):

            self._p.disconnect()


    def render(self, mode='rgb_array'):
        self._render_callback(mode)
        if mode == "human":
            return np.array([])
        # TODO: check the way to render image
        rgb_array, mask = render_image(self._p,RENDER_WIDTH, RENDER_HEIGHT,
                                       self._view_matrix, self._proj_matrix)
        if mode == 'rgb_array':
            return rgb_array
        else:
            return rgb_array, mask

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def compute_reward(self, achieved_goal, desired_goal, info):
        raise NotImplementedError

    def _env_setup(self):
        pass

    def _get_obs(self):
        raise NotImplementedError

    def _set_action(self, action):
        """ Applies the given action to the simulation.
        """
        raise NotImplementedError

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError

    def _sample_goal(self):
        """ Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _sample_goal_callback(self):
        """ For goal visualization, etc.
        """
        pass

    def _render_callback(self, mode):
        """ A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """ A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    @property
    def action_size(self):
        raise NotImplementedError

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a scripted oracle strategy
        """
        raise NotImplementedError

    def test(self, horizon=100):
        """
        Run the test simulation without any learning algorithm for debugging purposes
        """
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

    def __del__(self):
        self.close()

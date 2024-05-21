################################gym env的源代码#######################
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from pybullet_envs.bullet import kuka
import random
import pybullet_data
from pkg_resources import parse_version

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class KukaGymEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False,
               maxSteps=1000):
    #print("KukaGymEnv __init__")
    self._isDiscrete = isDiscrete
    self._timeStep = 1. / 240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._maxSteps = maxSteps
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180
    self._cam_pitch = -40

    self._p = p
    if self._renders:
      cid = p.connect(p.SHARED_MEMORY)
      if (cid < 0):
        cid = p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
    else:
      p.connect(p.DIRECT)
    #timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
    self.seed()
    self.reset()
    observationDim = len(self.getExtendedObservation())
    #print("observationDim")
    #print(observationDim)

    observation_high = np.array([largeValObservation] * observationDim)
    if (self._isDiscrete):
      self.action_space = spaces.Discrete(7)
    else:
      action_dim = 3
      self._action_bound = 1
      action_high = np.array([self._action_bound] * action_dim)
      self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(-observation_high, observation_high)
    self.viewer = None

  def reset(self):
    #print("KukaGymEnv _reset")
    self.terminated = 0
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

    p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
               0.000000, 0.000000, 0.0, 1.0)

    xpos = 0.55 + 0.12 * random.random()
    ypos = 0 + 0.2 * random.random()
    ang = 3.14 * 0.5 + 3.1415925438 * random.random()
    orn = p.getQuaternionFromEuler([0, 0, ang])
    self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), xpos, ypos, -0.15,
                               orn[0], orn[1], orn[2], orn[3])

    p.setGravity(0, 0, -10)
    self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0
    p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

  def __del__(self):
    p.disconnect()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self):
    self._observation = self._kuka.getObservation()
    gripperState = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaGripperIndex)
    gripperPos = gripperState[0]
    gripperOrn = gripperState[1]
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)

    invGripperPos, invGripperOrn = p.invertTransform(gripperPos, gripperOrn)
    gripperMat = p.getMatrixFromQuaternion(gripperOrn)
    dir0 = [gripperMat[0], gripperMat[3], gripperMat[6]]
    dir1 = [gripperMat[1], gripperMat[4], gripperMat[7]]
    dir2 = [gripperMat[2], gripperMat[5], gripperMat[8]]

    gripperEul = p.getEulerFromQuaternion(gripperOrn)
    #print("gripperEul")
    #print(gripperEul)
    blockPosInGripper, blockOrnInGripper = p.multiplyTransforms(invGripperPos, invGripperOrn,
                                                                blockPos, blockOrn)
    projectedBlockPos2D = [blockPosInGripper[0], blockPosInGripper[1]]
    blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)
    #print("projectedBlockPos2D")
    #print(projectedBlockPos2D)
    #print("blockEulerInGripper")
    #print(blockEulerInGripper)

    #we return the relative x,y position and euler angle of block in gripper space
    blockInGripperPosXYEulZ = [blockPosInGripper[0], blockPosInGripper[1], blockEulerInGripper[2]]

    #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir0[0],gripperPos[1]+dir0[1],gripperPos[2]+dir0[2]],[1,0,0],lifeTime=1)
    #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir1[0],gripperPos[1]+dir1[1],gripperPos[2]+dir1[2]],[0,1,0],lifeTime=1)
    #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir2[0],gripperPos[1]+dir2[1],gripperPos[2]+dir2[2]],[0,0,1],lifeTime=1)

    self._observation.extend(list(blockInGripperPosXYEulZ))
    return self._observation

  def step(self, action):
    if (self._isDiscrete):
      dv = 0.005
      dx = [0, -dv, dv, 0, 0, 0, 0][action]
      dy = [0, 0, 0, -dv, dv, 0, 0][action]
      da = [0, 0, 0, 0, 0, -0.05, 0.05][action]
      f = 0.3
      realAction = [dx, dy, -0.002, da, f]
    else:
      #print("action[0]=", str(action[0]))
      dv = 0.005
      dx = action[0] * dv
      dy = action[1] * dv
      da = action[2] * 0.05
      f = 0.3
      realAction = [dx, dy, -0.002, da, f]
    return self.step2(realAction)

  def step2(self, action):
    for i in range(self._actionRepeat):
      self._kuka.applyAction(action)
      p.stepSimulation()
      if self._termination():
        break
      self._envStepCounter += 1
    if self._renders:
      time.sleep(self._timeStep)
    self._observation = self.getExtendedObservation()

    #print("self._envStepCounter")
    #print(self._envStepCounter)

    done = self._termination()
    npaction = np.array([
        action[3]
    ])  #only penalize rotation until learning works well [action[0],action[1],action[3]])
    actionCost = np.linalg.norm(npaction) * 10.
    #print("actionCost")
    #print(actionCost)
    reward = self._reward() - actionCost
    #print("reward")
    #print(reward)

    #print("len=%r" % len(self._observation))

    return np.array(self._observation), reward, done, {}

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])

    base_pos, orn = self._p.getBasePositionAndOrientation(self._kuka.kukaUid)
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                            distance=self._cam_dist,
                                                            yaw=self._cam_yaw,
                                                            pitch=self._cam_pitch,
                                                            roll=0,
                                                            upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                     nearVal=0.1,
                                                     farVal=100.0)
    (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                              height=RENDER_HEIGHT,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
    #renderer=self._p.ER_TINY_RENDERER)

    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def _termination(self):
    #print (self._kuka.endEffectorPos[2])
    state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
    actualEndEffectorPos = state[0]

    #print("self._envStepCounter")
    #print(self._envStepCounter)
    if (self.terminated or self._envStepCounter > self._maxSteps):
      self._observation = self.getExtendedObservation()
      return True
    maxDist = 0.005
    closestPoints = p.getClosestPoints(self._kuka.trayUid, self._kuka.kukaUid, maxDist)

    if (len(closestPoints)):  #(actualEndEffectorPos[2] <= -0.43):
      self.terminated = 1

      #print("terminating, closing gripper, attempting grasp")
      #start grasp and terminate
      fingerAngle = 0.3
      for i in range(100):
        graspAction = [0, 0, 0.0001, 0, fingerAngle]
        self._kuka.applyAction(graspAction)
        p.stepSimulation()
        fingerAngle = fingerAngle - (0.3 / 100.)
        if (fingerAngle < 0):
          fingerAngle = 0

      for i in range(1000):
        graspAction = [0, 0, 0.001, 0, fingerAngle]
        self._kuka.applyAction(graspAction)
        p.stepSimulation()
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        if (blockPos[2] > 0.23):
          #print("BLOCKPOS!")
          #print(blockPos[2])
          break
        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        actualEndEffectorPos = state[0]
        if (actualEndEffectorPos[2] > 0.5):
          break

      self._observation = self.getExtendedObservation()
      return True
    return False

  def _reward(self):

    #rewards is height of target object
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
    closestPoints = p.getClosestPoints(self.blockUid, self._kuka.kukaUid, 1000, -1,
                                       self._kuka.kukaEndEffectorIndex)

    reward = -1000

    numPt = len(closestPoints)
    #print(numPt)
    if (numPt > 0):
      #print("reward:")
      reward = -closestPoints[0][8] * 10
    if (blockPos[2] > 0.2):
      reward = reward + 10000
      print("successfully grasped a block!!!")
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #print("reward")
      #print(reward)
    #print("reward")
    #print(reward)
    return reward

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step


from ..from_surrol.base_env.surrol_goalenv import SurRoLGoalEnv
from ..from_surrol.utils.pybullet_utils import reset_camera
from ..from_surrol.utils.pybullet_utils import get_link_pose, wrap_angle
#################################尝试直接改变环境代码##############
class KukaGraspEnv(SurRoLGoalEnv):
        #surol自带的3d模型
        ASSET_DIR_PATH="/home/aren/PYTHON/PYBULLET_SIM/long_horizon_chainning/SIM_env/sim_env/from_surrol/3d_model/from_surrol/"
        #控制模型大小吧 估计是
        SCALING = 1.
        #############必须自己实现的
        #1 初始化 基本采用原始代码的初始化流程
        def __init__(self,):

                #1设置渲染模式 开启bullet服务器
                super().__init__(render_mode='human')
                #2 相机参数设置 无需修正
                #3 种子设置--源代码中没有真正启用 gym的 torch的 numpy的 bullet的估计不用

                #4 bullet ui 搜索路径等服务参数配置
                #5 物理仿真参数配置 加载地板 objid字典创建 
                #6 场景设置--修改_env_setup

                #7 初始goal拿到 self._sample_goal和self._sample_goal_callback()
                #8 观察和动作空间设置 _get_obs()  保证是dict才行
                #9 仿真时间间隙 无需修正
        #2 初始化流程中的场景设置
        #2024/5/21场景设置完成 父类的 self.goal = self._sample_goal()之前认为功能是正常的       
        def _env_setup(self):
                #1 设置相机
                        
                        # camera
                        if self._render_mode == 'human':
                                reset_camera(yaw=90.0, pitch=-30.0, dist=2 * self.SCALING,
                                                target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))
        
                #2 部署机器人和场景物体
                        #1 封装好的kuka机器人
                        self._urdfRoot=pybullet_data.getDataPath()
                        self._timeStep = 1. / 240.
                        self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
                        #2 桌子
                        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
                                0.000000, 0.000000, 0.0, 1.0)
                        #3  随机扔几个物块
                        xpos = 0.55 + 0.12 * random.random()
                        ypos = 0 + 0.2 * random.random()
                        ang = 3.14 * 0.5 + 3.1415925438 * random.random()
                        orn = p.getQuaternionFromEuler([0, 0, ang])
                        self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), xpos, ypos, -0.15,
                                                orn[0], orn[1], orn[2], orn[3])
                        self.obj_ids['rigid'].append(self.blockUid)
        
                #3 显式路点的那个小红点

                        obj_id = p.loadURDF(os.path.join(self.ASSET_DIR_PATH, 'sphere/sphere.urdf'),
                                        globalScaling=self.SCALING)
                        self.obj_ids['fixed'].append(obj_id)  # 0
        #3 获取多种最终目标
        def _sample_goal(self) -> np.ndarray:
                #1 逻辑上来看 是物体的最终要达到的位置
                goal = np.array(get_link_pose(self.obj_ids['rigid'][0], -1))
                return goal.copy()
        #4 初始化流程中的子目标（路点）设置
        def _sample_goal_callback(self):
                #1 把之前没用到的sphere设置位置 为了可视化显式路点
                p.resetBasePositionAndOrientation(self.obj_ids['fixed'][0], self.goal, (0, 0, 0, 1))
                #2 路点的容器--列表 要多少个就给多少个
                self._waypoints = [None, None, None, None, None, None, None] 
                #3 路点怎么设置？
        #5 有了路点之后 就可以设置专家策略
        #goal引导的动作
        def get_oracle_action(self, obs) -> np.ndarray:
                pass
        #6 专家策略的动作 进到机械臂执行
        def _set_action(self, action: np.ndarray):
                pass
        #6 动作执行之后 得到纯观察
        def _get_obs(self) -> dict:
                pass
        #7 以上两个核心动作执行完之后  进行封装 经典的env.step()
        def step(self, action: np.ndarray):
                pass
        #7 另一个经典的gym接口 env.reset()
        def reset(self):
                pass

        def _is_success(self, achieved_goal, desired_goal):
                pass

        def _meet_contact_constraint_requirement(self):
                pass

        def subgoal_grasp(self):
                pass
        
        def is_success(self, achieved_goal, desired_goal):
                pass
        #@property各种需要的property
        #以下是基类的东西
        def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
                pass
        # def _env_setup(self):
        #         pass
        def _get_robot_state(self, idx: int) -> np.ndarray:
                pass

        def _is_success(self, achieved_goal, desired_goal):
                pass
       
        def _step_callback(self):
                pass
        #被子类覆盖
        def _sample_goal(self) -> np.ndarray:
                pass
        #被子类覆盖

        def _activate(self, idx: int):
                pass
        def _release(self, idx: int):
                pass
        #被子类覆盖
        def _meet_contact_constraint_requirement(self) -> bool:
                pass
        #goal env的东西



        #最初始的env的东西：
        #实现的
        #env成功的标志：过程中返回的奖励ok 专家策略能执行成功
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
                
        def seed(self, seed=None):
                pass
        def render(self, mode='rgb_array'):
                pass
        def close(self):
                pass
        #被子类覆盖
        def reset(self):
                pass
        #未实现
        def get_oracle_action(self, obs) -> np.ndarray:
                pass
        def _step_callback(self):
                pass
        def _render_callback(self, mode):
                pass
        def _sample_goal_callback(self):
                pass
        def _sample_goal(self):
                pass
        def _is_success(self, achieved_goal, desired_goal):
                pass
        def _set_action(self, action):
                pass
        def _get_obs(self):
                pass
        # def _env_setup(self):
        #         pass
        def compute_reward(self, achieved_goal, desired_goal, info):
                pass

        #########采用原版代码的
        def close(self):
                super().reset()
                print("原版的close")


##############原始的环境测试代码
if __name__ == "__main__":
    env = KukaGraspEnv()  # create one process and corresponding env

#     env.test()
#     env.close()
#     time.sleep(2)

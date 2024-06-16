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

from ..from_surrol.base_env.surrol_goalenv import SurRoLGoalEnv
from ..from_surrol.utils.pybullet_utils import reset_camera
from ..from_surrol.utils.pybullet_utils import get_link_pose, wrap_angle
#################################尝试直接改变环境代码##############
class PandaGraspEnv(SurRoLGoalEnv):
        #surol自带的3d模型
        ASSET_DIR_PATH="/home/aren/PYTHON/PYBULLET_SIM/long_horizon_chainning/SIM_env/sim_env/from_surrol/3d_model/from_surrol/"
        #控制模型大小吧 估计是
        SCALING = 1.
        #############必须自己实现的
        #1 初始化 基本采用原始代码的初始化流程
        def __init__(self,):
                #父类的逻辑
                #1设置渲染模式 开启bullet服务器
                #2 相机参数设置 无需修正
                #3 种子设置--源代码中没有真正启用 gym的 torch的 numpy的 bullet的估计不用
                #4 bullet ui 搜索路径等服务参数配置
                #5 物理仿真参数配置 加载地板 objid字典创建 
                #6 场景设置--修改_env_setup
                #7 初始goal拿到 self._sample_goal和self._sample_goal_callback()
                #8 观察和动作空间设置 _get_obs()  保证是dict才行
                #9 仿真时间间隙 无需修正
                p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)

                super().__init__(render_mode='human')
        
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
                        offset=[0,0,0]
                        self.offset = np.array(offset)
                        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                        self.legos=[]
                        p.loadURDF("tray/traybox.urdf", [0+offset[0], 0+offset[1], -0.6+offset[2]], [-0.5, -0.5, -0.5, 0.5], flags=flags)
                        self.legos.append(p.loadURDF("lego/lego.urdf",np.array([0.1, 0.3, -0.5])+self.offset, flags=flags))
                        p.changeVisualShape(self.legos[0],-1,rgbaColor=[1,0,0,1])
                        self.legos.append(p.loadURDF("lego/lego.urdf",np.array([-0.1, 0.3, -0.5])+self.offset, flags=flags))
                        self.legos.append(p.loadURDF("lego/lego.urdf",np.array([0.1, 0.3, -0.7])+self.offset, flags=flags))
                        self.sphereId = p.loadURDF("sphere_small.urdf",np.array( [0, 0.3, -0.6])+self.offset, flags=flags)
                        p.loadURDF("sphere_small.urdf",np.array( [0, 0.3, -0.5])+self.offset, flags=flags)
                        p.loadURDF("sphere_small.urdf",np.array( [0, 0.3, -0.7])+self.offset, flags=flags)
                        orn=[-0.707107, 0.0, 0.0, 0.707107]#p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
                        eul = p.getEulerFromQuaternion([-0.5, -0.5, -0.5, 0.5])
                        self.panda = p.loadURDF("franka_panda/panda.urdf", np.array([0,0,0])+self.offset, orn, useFixedBase=True, flags=flags)
                        import pdb;pdb.set_trace()

                        
                        #self.obj_ids['rigid'].append(self.blockUid)
        
                #3 显式路点的那个小红点

                        obj_id = p.loadURDF(os.path.join(self.ASSET_DIR_PATH, 'sphere/sphere.urdf'),
                                        globalScaling=self.SCALING)
                        self.obj_ids['fixed'].append(obj_id)  # 0
                        self.panda_action_control_test()                        

                        
        #[位置变化，旋转变化，执行器控制] 这是机械臂控制的确定的东西
        def panda_action_control_test(self,action=[1,1,1]):
                #拿到action
                #有个问题 action是什么东西？和论文源代码的控制方式差距有多大？
                #结论 差距不大 全都是末端执行器的位姿 xyz三个 加上旋转说不清几个  然后执行器开关一个 五个六个反正
                dv = 0.005
                dx = action[0] * dv
                dy = action[1] * dv
                da = action[2] * 0.05
                f = 0.3
                realAction = [0.001, 0, 0, 0, 0.5]
                #传进机械臂
                while True:
                        import random
                        realAction = [0, 0, 0, 0, 0]
                        realAction[2] = random.uniform(-0.0001, 0.0001)
                        self._kuka.applyAction(realAction)
                        p.stepSimulation()
                #测试机械臂的obs获取        

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

        #5  进行封装 经典的env.step()
        def step(self, action: np.ndarray):
                #父类的逻辑
                #1 动作归一化
                #2 set action进机器人
                #3 物理步进
                #4 
                #5 拿obs
                #6 计算奖励和其他信息 封装为旧gym的四元素obs,reward,done,info
                obs, reward, done, info=super().step(action)
                return obs, reward, done, info

        def _set_action(self, action: np.ndarray):
                #转为列表 kukagym的代码接口是列表
                action=action.tolist()
                dv = 0.005
                dx = action[0] * dv
                dy = action[1] * dv
                da = action[2] * 0.05
                f = 0.3
                realAction = [dx, dy, -0.002, da, f]
                self._actionRepeat=1
                for i in range(self._actionRepeat):
                        self._kuka.applyAction(realAction)
                        p.stepSimulation()
                
                #有个问题 termination怎么回事？为什么goalenv没有termination的函数？
                #专门额外操控gripper 可能认为是到了近的位置了 才开始抓
                state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
                actualEndEffectorPos = state[0]
                maxDist = 0.005
                closestPoints = p.getClosestPoints(self._kuka.trayUid, self._kuka.kukaUid, maxDist)
                if (len(closestPoints)):
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
  
        #怀疑是激活抓取
        def _activate(self, idx: int):
                pass
       
        #做出抓的动作之后 让物体粘在末端执行器上面 模拟力封闭抓取             
        def _step_callback(self):
                
                pass
        
        def _meet_contact_constraint_requirement(self) -> bool:
                pass
        
        def _release(self, idx: int):
                pass
  

        def _get_obs(self) -> dict:
                #来自kuka gym  拿纯观察
                self._observation = []
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
        #拿机器人状态
        def _get_robot_state(self, idx: int) -> np.ndarray:
              pass  

        def _is_success(self, achieved_goal, desired_goal):
                pass                
        
        def is_success(self, achieved_goal, desired_goal):
                
                return self._is_success(achieved_goal, desired_goal)
       
        def compute_reward(self, achieved_goal, desired_goal, info):
                pass

        #6 另一个经典的gym接口 env.reset()
        def reset(self):
                #父类的逻辑
                #1 模拟关 重力设置 ui关
                #2 重新load场景 并且步进
                #3 重新拿goal
                #4 开ui
                #5 拿obs
                super().reset()
        
        #7 关仿真 没啥说的
        def close(self):
                super().close()

        #env成功的标志：过程中返回的奖励ok 专家策略能执行成功
        def test(self, horizon=100):
                steps, done = 0, False
                #1 reset能成功 
                obs = self.reset()
                while not done and steps <= horizon:
                        tic = time.time()
                        #2 示教动作能拿到
                        action = self.get_oracle_action(obs)
                        print('\n -> step: {}, action: {}'.format(steps, np.round(action, 4)))
                        # print('action:', action)
                        #3 step能成功
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

        #没有在主逻辑看见调用 可以认为是测试用 测试子目标设计的有没有问题 能不能成功执行抓取
        def subgoal_grasp(self):
                pass
        
        #@property各种需要的property
        #以下是基类的东西

        def seed(self, seed=None):
                pass
        
        #渲染图片的  记录的时候需要
        def render(self, mode='rgb_array'):
                pass

        def _render_callback(self, mode):
                pass
        
   






##############原始的环境测试代码
if __name__ == "__main__":
    env = PandaGraspEnv()  # create one process and corresponding env

#     env.test()
#     env.close()
#     time.sleep(2)

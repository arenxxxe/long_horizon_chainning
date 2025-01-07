import time
import numpy as np
import math
from pybullet_utils import bullet_client as bc
import pybullet
import pybullet_data as pd

useNullSpace = 1
ikSolver = 0
#末端执行器的id？
pandaEndEffectorIndex = 11 #8
pandaNumDofs = 7

ll = [-7]*pandaNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs
#restposes for null space
jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66,-0.79, 0.02, 0.02]
#最后两个是夹爪的直线轴  9和1- 还有个看不见的eelink 这是末端位置的轴 应该动不了 所以u估计是8轴了 控制旋转的
#前面是0-6的轴 估计是和末端无关的  
# jointPositions=[0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00]
rp = jointPositions 
from memory_profiler import profile
class PandaArm(object):
       
#   @profile(stream=open('grasp--test----panda初始化的泄漏问题.txt','w'))

  def __init__(self, bullet_client, offset):
    self.bullet_client = bullet_client
    self.offset = np.array(offset)
    #print("offset=",offset)

    flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

    # legos=[]
    # self.bullet_client.loadURDF("tray/traybox.urdf", [0+offset[0], 0+offset[1], -0.6+offset[2]], [-0.5, -0.5, -0.5, 0.5], flags=flags)
    # legos.append(self.bullet_client.loadURDF("lego/lego.urdf",np.array([0.1, 0.3, -0.5])+self.offset, flags=flags))
    # legos.append(self.bullet_client.loadURDF("lego/lego.urdf",np.array([-0.1, 0.3, -0.5])+self.offset, flags=flags))
    # legos.append(self.bullet_client.loadURDF("lego/lego.urdf",np.array([0.1, 0.3, -0.7])+self.offset, flags=flags))
    # sphereId = self.bullet_client.loadURDF("sphere_small.urdf",np.array( [0, 0.3, -0.6])+self.offset, flags=flags)
    # self.bullet_client.loadURDF("sphere_small.urdf",np.array( [0, 0.3, -0.5])+self.offset, flags=flags)
    # self.bullet_client.loadURDF("sphere_small.urdf",np.array( [0, 0.3, -0.7])+self.offset, flags=flags)
    ##################尝试修改四元数 沿着z轴#############
    #orn=[-0.707107, 0.0, 0.0, 0.707107]#p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])

    orn = self.bullet_client.getQuaternionFromEuler([ 0,0, 0])
    eul = self.bullet_client.getEulerFromQuaternion([-0.5, -0.5, -0.5, 0.5])
    
    self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0,0,0])+self.offset, orn, useFixedBase=True, flags=flags)
    
    index = 0
    for j in range(self.bullet_client.getNumJoints(self.panda)):
      self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
      info = self.bullet_client.getJointInfo(self.panda, j)
  
      jointName = info[1]

    
      jointType = info[2]

      if (jointType == self.bullet_client.JOINT_PRISMATIC):
        
        self.bullet_client.resetJointState(self.panda, j, jointPositions[index]) 
        index=index+1
      if (jointType == self.bullet_client.JOINT_REVOLUTE):
        self.bullet_client.resetJointState(self.panda, j, jointPositions[index]) 
        index=index+1

    # self.bullet_client.resetBasePositionAndOrientation(self.panda, [-0.000000, 0.000000, 0.070000],
    #                                   [0.000000, 0.000000, 0.000000, 1.000000])

    pos = [0.6320828071580811 ,-0.001113897653747268 ,0.1]
    orn = self.bullet_client.getQuaternionFromEuler([0,-math.pi,math.pi])

    for i in range(100):
        jointPoses = self.bullet_client.calculateInverseKinematics(self.panda,pandaEndEffectorIndex, pos, orn, ll, ul,
      jr, rp, maxNumIterations=5)
        #0-6的轴在动而已
        for i in range(pandaNumDofs):

            self.bullet_client.resetJointState(self.panda, i, jointPoses[i]) 
        self.bullet_client.stepSimulation()
        
    self.t = 0.


  def get_ee_state(self):

        finger1 = self.bullet_client.getJointState(self.panda, 9)[0]
        finger2 = self.bullet_client.getJointState(self.panda, 10)[0]

        fingers_width = finger1+finger2

        pos=self.bullet_client.getLinkState(self.panda,11)[0]
        # print(f"末端执行器位置::::{pos}")
        return pos+(fingers_width,)

  def reset(self):
            
        index = 0
        for j in range(self.bullet_client.getNumJoints(self.panda)):
                self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
                info = self.bullet_client.getJointInfo(self.panda, j)
                
                jointName = info[1]

                
                jointType = info[2]

                if (jointType == self.bullet_client.JOINT_PRISMATIC):
                        
                        self.bullet_client.resetJointState(self.panda, j, jointPositions[index]) 
                        index=index+1
                if (jointType == self.bullet_client.JOINT_REVOLUTE):
                        self.bullet_client.resetJointState(self.panda, j, jointPositions[index]) 
                        index=index+1
        pos = [0.6320828071580811 ,-0.001113897653747268 ,0.1]
        orn = self.bullet_client.getQuaternionFromEuler([0,-math.pi,math.pi])

        for i in range(100):
                jointPoses = self.bullet_client.calculateInverseKinematics(self.panda,pandaEndEffectorIndex, pos, orn, ll, ul,
        jr, rp, maxNumIterations=5)
                #0-6的轴在动而已
                for i in range(pandaNumDofs):

                        self.bullet_client.resetJointState(self.panda, i, jointPoses[i]) 
                        self.bullet_client.stepSimulation()

        self.t = 0.


  def step(self,ee_action:np.ndarray):
        t = self.t
        self.t += 1./60.
        pos = [ee_action[0],ee_action[1],ee_action[2]]
        # print(f"机械臂一直执行的动作::::{pos}")

        orn = self.bullet_client.getQuaternionFromEuler([0,-math.pi,math.pi])
        jointPoses = self.bullet_client.calculateInverseKinematics(self.panda,pandaEndEffectorIndex, pos, orn, ll, ul,
        jr, rp, maxNumIterations=200,residualThreshold=1e-6)

        for i in range(pandaNumDofs):
                # self.bullet_client.resetJointState(self.panda, i, jointPoses[i]) 
                self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, jointPoses[i],force=5 * 240.)
        # while True:
        #         #末端的两个夹爪       
        #         self.bullet_client.setJointMotorControl2(self.panda, 9, self.bullet_client.POSITION_CONTROL, ee_action[3]/2,force=5 * 240.)
        #         self.bullet_client.setJointMotorControl2(self.panda, 10, self.bullet_client.POSITION_CONTROL, ee_action[3]/2,force=5 * 240.)
        #         #step测试一下夹爪的开合
        #         for i in range(100):
        #                 self.bullet_client.stepSimulation()
        #         #末端的两个夹爪       
        #         self.bullet_client.setJointMotorControl2(self.panda, 9, self.bullet_client.POSITION_CONTROL, 0,force=5 * 240.)
        #         self.bullet_client.setJointMotorControl2(self.panda, 10, self.bullet_client.POSITION_CONTROL, 0,force=5 * 240.)
        #         for i in range(100):
        #                 self.bullet_client.stepSimulation()

        # pass
        self.bullet_client.setJointMotorControl2(self.panda, 9, self.bullet_client.POSITION_CONTROL, ee_action[3]/2,force=5 * 240.)
        self.bullet_client.setJointMotorControl2(self.panda, 10, self.bullet_client.POSITION_CONTROL, ee_action[3]/2,force=5 * 240.)
        def reset(self):
               pass


####################################################################使用方法#############################################################

import math
import time
import numpy as np

# def main():
    
#     p.connect(p.GUI)
    

#     p.setAdditionalSearchPath(pd.getDataPath())

#     timeStep=1./60.
#     p.setTimeStep(timeStep)
#     ##########################改z轴##########################
#     p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,0)
#     p.setGravity(0, 0, -9.8)

    
#     panda = PandaSim(p,[0,0,0])
#     while (1):
#         panda.step()
#         p.stepSimulation()
#         time.sleep(timeStep)
	
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
import random
from pkg_resources import parse_version

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


from memory_profiler import profile

from chaining_package.ENV.base_env.viskill_base_env.surrol_goalenv import SurRoLGoalEnv
def reset_camera(bullet_client,yaw=50.0, pitch=-35.0, dist=5.0, target=(0.0, 0.0, 0.0)):
    bullet_client.resetDebugVisualizerCamera(
        cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=target)
#################################尝试直接改变环境代码################################
pybullet_wierd_offset=0.02
class PandaGraspEnv(SurRoLGoalEnv):
        #surol自带的3d模型
        ASSET_DIR_PATH=os.path.abspath("./chaining_package/ENV/3d_asset/3d_model/kuka_grasp")
        
        #调整各种位置和大小的东西 其实没啥用处
        ee_offset=0.255
        SCALING = 1.
        #A 完成上层调用逻辑     env = KukaGraspEnv() 
        #TODO 已经完成书写
        def __init__(self,render_mode='rgb_array',state_save_id=0):
                

                self.grasp_or_release =True
                self._panda=None
                self.stacked_block=None
                self.red_obj_id =None
                self.stacked_block=None
                self._p=None
                self.state_save_id=state_save_id
                super().__init__(render_mode=render_mode)

        @property
        def action_size(self):
                return 4 #xyz 末端执行器关节旋转 +夹爪开合
        
        # @profile(stream=open('最底层的grasp的env 的env steup泄漏.txt','w'))

        def _env_setup(self): 
                """
                处于的流程: A-6 场景设置
                输入：无
                输出：无
                目的：设置相机  部署机器人和场景物体  导入显式路点的那个小红点模型
                TODO 基本已经完成
                """


                #1 设置相机



                                # camera
                #改变角度拍摄更加清楚
                if self._render_mode == "rgb_array":
                        reset_camera(self._p,yaw=90.0, pitch=-5.0, dist=2 * self.SCALING,
                                        target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))
                if self._render_mode == "human":
                        reset_camera(self._p,yaw=90.0, pitch=-5.0, dist=2 * self.SCALING,
                                target=(-0.05 * self.SCALING-0.3, 0, 0.36 * self.SCALING))
                        #2c  随机扔几个物块
                self.init_block_xpos = 0.6320828071580811  
                self.init_block_ypos = -0.001113897653747268  
                ang = 3.14 * 0.5 + 3.1415925438 * 0.5
                orn = self._p.getQuaternionFromEuler([0, 0, ang])
                self.init_stacked_block_xpos=self.init_block_xpos+0.12 * 0.5
                self.init_stacked_block_ypos=self.init_block_ypos+0.2 * 0.8
                           


                # if  self._panda is None:
                # print("触发创建")

                #2 部署机器人和场景物体
                #2a 封装好的kuka机器人
                self._urdfRoot=os.path.abspath("./chaining_package/ENV/3d_asset/3d_model/kuka_grasp")
                
                self._timeStep = 1. / 240.

                #去掉2 
                # p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -0.63000,
                #         0.000000, 0.000000, 0.0, 1.0)
                
                #机械臂抓的
                self.blockUid = self._p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"),[self.init_block_xpos, self.init_block_ypos, 0.014],
                                        [orn[0], orn[1], orn[2], orn[3]])
                #被抓的
                self.stacked_block= self._p.loadURDF(os.path.join(self._urdfRoot, "blue_block.urdf"), [self.init_stacked_block_xpos , self.init_stacked_block_ypos ,0.014],
                        [orn[0], orn[1], orn[2], orn[3]])
                
                        #去掉1 
                self._panda = PandaArm(self._p,[0,0,0])
                self.pandauid=self._panda.panda


                #self.kuka_body=self._kuka.kukaUid

                self.obj_ids['rigid'].append(self.blockUid)

                #3 导入显式路点的那个小红点模型
                self.red_obj_id = self._p.loadURDF(os.path.join(self._urdfRoot, 'sphere/sphere.urdf'),
                                globalScaling=self.SCALING*10)
                self.obj_ids['fixed'].append(self.red_obj_id)  # 0


                        #简单测试区域 前面的东西认为是没问题的
                        #直接拿上个动作
                        #self.last_ee_action=self._kuka.last_action
                        # self.goal=self._sample_goal()
                        # self._sample_goal_callback()
                        #self.kuka_action_control_test()
                # else:
                #         # print("触发reset")
                #         #机械臂抓的
                #         p.resetBasePositionAndOrientation(self.blockUid, [self.init_block_xpos, self.init_block_ypos, 0.014], (orn[0], orn[1], orn[2], orn[3]))
                #         #被堆叠的
                #         p.resetBasePositionAndOrientation(self.stacked_block, [self.init_stacked_block_xpos , self.init_stacked_block_ypos ,0], (orn[0], orn[1], orn[2], orn[3]))
                         
                #         self._panda.reset()
                self._p.saveBullet(f"{self.state_save_id}_state.bullet")
                self._p.stepSimulation()





                        
        def _sample_goal(self):
                """
                处于的流程: A-7 初始goal拿到
                输入：无
                输出：无
                目的: 设置env传出外面的goal 最终物体要到的位置
                
                TODO 基本已经完成
                goal--物体最终落地之后的位置 
                """
                #改变了之后 物体的位置也是会改变的 
                goal = [ 0.6919, 0.159 ,0.0700 ]
                np_goal=np.array(goal)
                return np_goal.copy()
        
        def _sample_goal_callback(self):
                """
                处于的流程: A-8 示教策略和subgoal设置
                输入：无
                输出：无
                目的: 设置整体运动规划路点 设置subgoal(物体大变化的位置)
                
                TODO 基本已经完成
                list: self._waypoints 示教策略
                list: self.subgoals 物体移动子目标
                """
                #1 路点可视化 用导入的红色圆球 注意scaling大一点 不然看不到 可以搞多个

                #print(self.obj_ids)

                red_point_init_pos=[self.init_block_xpos,self.init_block_ypos,0.1]

                self._p.resetBasePositionAndOrientation(self.obj_ids['fixed'][0], red_point_init_pos, (0, 0, 0, 1))
                
                #2 设置路点（末端位姿和动作）--示教策略
                        #原作者代码的思路：
                        # 拿到起点、中间位置、终点的xyz坐标
                        # 拿到物体和末端执行器的旋转
                        # 末端执行器旋转归一化
                        # 路点加噪声
                        # 切路点为子任务 切分逻辑： 接近算是一个目标 做动作又是一个目标
                                # 5.1 抓取目标：在物体上方--接近物体---末端夹爪动作---抬起物体到指定位置
                                # 5.2 转移：接近一个放 一个夹
                                # 5.3 peg：在上方---释放
                
                


                init_object_posx=self.init_block_xpos
                init_object_posy=self.init_block_ypos
                init_object_posz=self._p.getBasePositionAndOrientation(self.blockUid)[0][2]
                init_above_object_posz=0.1


                #噪声

                #noise_vector=[-0.0022,0.0119,0.0027]#起码这个点可用
                #范围 x：-0.0022 --- 0.0044
                #范围 y : 0.004 ----- 0.0119
                #范围 z : 0.0027 ---- 0.004
                # x_range=[-0.0022 ,0.0044]
                # y_range=[0.003,0.0110]
                # z_range=[0.0027,0.004]


                x_range=[-0.0022 ,0.0022]
                y_range=[0.003,0.0040]
                z_range=[0.0027,0.004]
                noise_vector=np.array([np.random.uniform(x_range[0],x_range[1]),
                                       np.random.uniform(y_range[0],y_range[1]),
                                       np.random.uniform(z_range[0],z_range[1])
                                       ])
                # debug
                # noise_vector=np.array([0,0,0])
                
                noise_vector_expand=np.pad(noise_vector,(0,1),"constant",constant_values=0)



                #print(f"此时的向
                # 量{noise_vector}")
                self.ee_open=0.08
                self.ee_close=0.05
                fake_open=0.5
                fake_close=-0.5
                # 0.5开 转1.5 -0.5 关 转0
                ####重要原则 一条路径上 两个路点只能有同一个xyz目标
                                #1.1 先定位
                above_object_wp=[init_object_posx,init_object_posy,init_above_object_posz,fake_open]
                above_object_wp+=noise_vector_expand

                reach_object_wp=[init_object_posx,init_object_posy,init_object_posz,fake_open]   
                reach_object_wp+=noise_vector_expand

                grasp_object_wp=[init_object_posx,init_object_posy,init_object_posz,fake_close]    
                grasp_object_wp+=noise_vector_expand

                lift_object_wp=[init_object_posx,init_object_posy,init_above_object_posz,fake_close]
                lift_object_wp+=noise_vector_expand

                move_object_wp=[self.init_stacked_block_xpos,self.init_stacked_block_ypos,init_above_object_posz,fake_close]
                move_object_wp+=noise_vector_expand

                release_object_wp=[ move_object_wp[0], move_object_wp[1], move_object_wp[2],fake_open]
                release_object_wp+=noise_vector_expand

                
                self._waypoints = [above_object_wp,reach_object_wp,grasp_object_wp,lift_object_wp,move_object_wp,release_object_wp] 
                
                grasp_object_goal=[grasp_object_wp[0],grasp_object_wp[1],grasp_object_wp[2]]
                # lift_object_goal=[lift_object_wp[0],lift_object_wp[1],lift_object_wp[2]]
                #这里之后应该是物体目标 因为此时的观察拿的是物体的位置做goal
                move_object_goal=[move_object_wp[0],move_object_wp[1],move_object_wp[2]]

                release_object_goal=[self.goal[0],self.goal[1],self.goal[2]]
                release_object_goal+=noise_vector_expand[0:3]

                self.subgoals=[grasp_object_goal,move_object_goal,release_object_goal]

                

                
        def _get_obs(self) :

                
                #1 机器人状态
                object_pos, _ = self._p.getBasePositionAndOrientation(self.blockUid)
                np_object_pos=np.array(object_pos)
                # np_object_pos=np.round(np_object_pos,4)

                # 拿到的是
                self.panda_ee_state = self._panda.get_ee_state()
                                #夹爪开合角度
                


                robot_state=np.array(self.panda_ee_state)
                        
                # robot_state=np.round(robot_state,4)

                def pairwise_collision(bullet_client,body1, body2, max_distance=0):  # 10000
                # getContactPoints
                # return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance)) != 0

                        return bullet_client.getContactPoints(body1, body2) != ()

                object_rel_pos1=np_object_pos - robot_state[0: 3]

                # np_object_rel_pos1=np.round(object_rel_pos1,4)

                
                # print(self.grasp_or_release)

                if self.grasp_or_release:
                #        print("机械臂末端")
                       achieved_goal=robot_state[0: 3]
                else:
                #        print("物体末端")
                       achieved_goal=np_object_pos


                observation = np.concatenate([
                robot_state, np_object_pos.ravel(), object_rel_pos1.ravel()
                ])


                #3 总观察
                obs = {
                'observation': observation.copy(),
                'achieved_goal': achieved_goal.copy(),
                'desired_goal': self.goal.copy()
                }
                # print(f"末端状态{obs['observation'][0: 3]}")
                return obs
        

                
        def step(self, action: np.ndarray):
                """
                处于的流程: B-3 传入action 完成一次仿真步进
                输入： list:acton
                输出：  dict:obs, reward, bool:done, info
                目的: 动作进来 完成一次仿真步进
                TODO  
                """
                #1 动作归一化 --动作限制在什么范围好？
                #2 机器人设置动作--set_action 直接在这个函数执行仿真步进
                #3 物理步进--step函数--从源代码基类注释掉了 不符合kuka机械臂的api
                #4 力封闭约束--step_callback--无视
                #5 拿obs--get_obs
                #6 done 一直是false
                #8 info仅判断是否到达目标 --self._is_success
                #9 计算奖励 --self.compute_reward
                obs, reward, done, info=super().step(action)
                return obs, reward, done, info

        def _set_action(self, action:np.ndarray):
                """
                处于的流程: B-3-2 机器人设置动作
                输入： np_array:acton
                输出：  
                目的: 进行一次完整的仿真步进成功执行步骤
                TODO  
                """
 
                # print(f"施教要搜集的动作————————》{action}")

                #init_object_pos=p.getBasePositionAndOrientation(self.obj_ids['rigid'][0])
                #print(f"物体位置{init_object_pos}")
                #拿之前的动作
                # self.last_ee_action=self._kuka.last_action
                #进来的action  是一个相对量 
                # eeobs,_=self._kuka.getEE_pos() #真实的末端位置
                # functor_ee_pos=np.array(eeobs)
                #开关
                if action[3]<0:
                        action[3]=self.ee_close
                elif action[3]>=0:
                        action[3]=self.ee_open

                # if self.state_save_id==1:
                #        print(action)
                action[:3]*=0.04#限制整体动作的大小量级 避免飘来飘去
                eeobs=self._panda.get_ee_state()
                
                functor_ee_pos=np.array(eeobs[0:3])
                # print(functor_ee_pos)
                # #进来的动作默认是-1到1  那么0.1的限制 使得动作只能在0.01的数量级
                ee_action=np.concatenate([functor_ee_pos+action[:3],action[-1:]])#得到现在真正要前进的位置
                # #print(f"action放缩{action[:3]*0.1}\n")
                #直接硬限制动作空间
                ee_action[:3]=np.clip(ee_action[:3],[0.6,-0.03,0.002],[0.7,0.3,0.2])

                ee_action=np.round(ee_action,4)


                #加上新的限制 0.01
                #ee_action[0:3]*=0.01
                assert len(ee_action) == self.action_size
                #储存上一个动作 为了0输入的动作
                #self.last_ee_action=[action[3],action[4]]
                # print(f"准备传进去的东西————————》{ee_action}")
                self._p.resetBasePositionAndOrientation(self.obj_ids['fixed'][0], [action[0],action[1],action[2]], (0, 0, 0, 1))
                for i in range(30):
                        self._panda.step(ee_action)
                        self._p.stepSimulation()
                        # if self.state_save_id==1:
                               
                        # time.sleep(1.0 /120.0)

 
                
           
        def _step_callback(self):
                #源代码中 用来进行模拟力封闭抓取的 思路就是使用pybullet的约束 直接把物体锁死在末端执行器上面 实现稳定抓取
                pass
           
        def goal_distance(self,goal_a, goal_b):
                
                #很明显的是 作了目标替换之后  计算奖励的逻辑出问题了
                assert goal_a.shape == goal_b.shape
                return np.linalg.norm(goal_a - goal_b, axis=-1)
        
        def _is_success(self, achieved_goal, desired_goal,threshold=0.006):
                """
                处于的流程: B-3-8 判断是否到达最终目标
                输入： np_array:achieved_goal, np_array:desired_goal
                输出：  bool
                目的: 比较现在目标和最终目标
                TODO  
                """

                self.distance_threshold=threshold
                d = self.goal_distance(achieved_goal, desired_goal)

                return (d < self.distance_threshold).astype(np.float32)                  

        def compute_reward(self, achieved_goal, desired_goal, info):

                return self._is_success(achieved_goal, desired_goal).astype(np.float32) - 1.
        
        # @profile(stream=open('最底层的grasp的env reset泄漏问题.txt','w'))

        def reset(self):
                """
                处于的流程: B-1 场景重置
                输入：无
                输出：无
                目的: 机械臂重置为初始位姿 物体重置为初始位姿 路径红点重置为初始
                TODO 需要进算法看一下输出是否符合算法要求格式
                """
                #1 模拟关 重力设置 ui关
                #2 重新load场景 并且步进 _env_setup()
                #3 重新拿goal _sample_goal()   _sample_goal_callback()
                #4 开ui
                
                #5 拿obs _get_obs()
                
                obs=super().reset()

                return obs
                
        def _get_robot_state(self) -> np.ndarray:

                #1 机器人状态
        
                self.kuka_ee_state = self._kuka.getObservation()
                                #夹爪开合角度
                _,joint_angle=self._kuka.getEE_pos()
                
                self.kuka_ee_state.extend([joint_angle])  #4+1             
                
                robot_state=np.array(self.kuka_ee_state)
                return robot_state

        def close(self):
                super().close()

        def get_oracle_action(self, obs) :
                """
                处于的流程: B-2  根据观察拿示教动作
                输入： dic:obs
                输出：  list:action
                目的: 根据观察 判断和路点的距离 根据距离 切换路点
                TODO 基本认为完成 如果不行 那就是距离判定的计算有问题 当然比较本质的肯定是信息的获取的准确的问题
                """
                action=[0]*4
                for i, waypoint in enumerate(self._waypoints):
                        #1 完成的路点跳过
                        if waypoint is None:
                                continue
                        #2 计算当前末端执行器位置和路点位置之间的距离

                        action=waypoint
                        np_waypoint=np.array(waypoint)
                        #3 计算观察中的东西和路点之间的距离--取自论文源代码
                        #print(f"路点{waypoint}")
                        #print(f"末端观察{obs['observation'][0: 3]}")
                        delta_pos=(waypoint[0: 3] - obs['observation'][0: 3]) /0.01 / 5.  

                        if np.abs(delta_pos).max() > 1:
                                delta_pos /= np.abs(delta_pos).max()

                        scale_factor = 0.4
                        delta_pos *= scale_factor 
                        #print(f"位置差{np.linalg.norm(delta_pos) * 0.01 / scale_factor}")
                        #print(f"旋转差{np.abs(delta_yaw)}  参考：{np.deg2rad(2.)}")
                        #4 判断是否到达位置 删路点
                        if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-3 :# 实际上三delta的最大值要小于0.05 这个阈值还是很大的
                                #print(f"第{i+1}个路点已经执行完毕")
                                if i == 5:
                                        #print("最后一个路点不会归零")
                                        pass                           
                                self._waypoints[i] = None if i< len(self._waypoints)-1 else self._waypoints[i]
                        #5 没到达位置 继续执行原来路点
                        break

                                # 进来的是期望走到的位姿态

                #目的： 切分微分路点 
                robot_and_ee=self._panda.get_ee_state()#真实的末端位置
                #print(f"eeobs{eeobs[2]}")
                functor_ee_pos=np.array(robot_and_ee)

                self.mimimun_delta_action_threshold=0.01#小于这个值就不要切分了 直接用delta action直接过去
                delta_ee_pos=action[0:3]-functor_ee_pos[0:3]

                # delta_ee_pos*=4 
                # #
                delta_ee_pos= np.clip(delta_ee_pos, self.action_space.low[0:3], self.action_space.high[0:3])

                for index,value in enumerate(delta_ee_pos):
                        if abs(value)<=self.mimimun_delta_action_threshold:
                                delta_ee_pos[index]=value*25
                        else :
                        #       delta_ee_pos[index]=self.mimimun_delta_action_threshold if value >= 0 else -self.mimimun_delta_action_threshold
                              delta_ee_pos[index]=value*0.2*25
                # delta_ee_pos= np.clip(delta_ee_pos, self.action_space.low[0:3], self.action_space.high[0:3])

                delta_action=np.concatenate([delta_ee_pos,action[-1:]])
                # factor_dv= 1
                #一系列放缩  施教的路点切分为一系列的小路点去执行
                factor_delta_action=delta_action



                return factor_delta_action,i
        

        def test(self, horizon=1000):
                """
                B 上层调用逻辑     env.test() 
                输入： dic:obs
                输出：  list:action
                目的: 机械臂重置为初始位姿 物体重置为初始位姿 路径红点重置为初始
                TODO 进行到step
                """
                #1 reset能成功 --self.reset()
                #2 根据观察拿示教动作 --self.get_oracle_action(obs)
                #3 传入action 完成一次仿真步进（大概60次仿真步进 ）--self.step(action)

                steps, done = 0, False
                
                obs = self.reset()

                while not done and steps <= horizon: #step特指每一次仿真步进 rollout或者episode指的是从头到尾完整执行一次仿真任务
                        #######################################进行panda 抓取精确性的测试###################
                        ee_action=np.array(self._waypoints[5])
                        
                        if ee_action[3]<0:
                                ee_action[3]=self.ee_close
                        elif ee_action[3]>=0:
                                ee_action[3]=self.ee_open
                        # for i in range(1000):
                        #         obs=self._get_obs()

                        #         self._panda.step(ee_action)
                        #         self._p.stepSimulation()
                        # breakpoint()
                        ##############################################################################
                        tic = time.time()
                        action,i = self.get_oracle_action(obs)
                        # if i==5 :
                        #         print(f"现在的路点steps是{steps}")
                        #         breakpoint()
                        print('\n -> step: {}, action: {}'.format(steps, np.round(action, 4)))

                        obs, reward, done, info = self.step(action)
                        # print(f"奖励{reward}")
                        if isinstance(obs, dict):
                                print(" -> achieved goal: {}".format(np.round(obs['achieved_goal'], 4)))
                                print(" -> desired goal: {}".format(np.round(obs['desired_goal'], 4)))
                        # else:
                        #         print(" -> achieved goal: {}".format(np.round(info['achieved_goal'], 4)))
                        done = info['is_success'] if isinstance(obs, dict) else done
                        if done:
                                print(f"wan的步长{steps}")
                                
                        steps += 1
                        toc = time.time()
                        #print(" -> step time: {:.4f}".format(toc - tic))

                        # #找最终的位置        
                        # object_pos, _ = self._p.getBasePositionAndOrientation(self.blockUid)
                        # np_object_pos=np.array(object_pos)
                        # print(f"物体最终位置{np_object_pos}")
                        # print('\n -> Done: {}\n'.format(done > 0))


        def seed(self, seed=None):
                pass
        #c 完成上层调用逻辑     env.render()

        def render(self,mode='rgb_array'):
                self._render_callback(mode)
                if mode == "human":
                        return np.array([])
                # TODO: check the way to render image
                rgb_array, mask = self.render_image(RENDER_WIDTH, RENDER_HEIGHT,
                                        self._view_matrix, self._proj_matrix)
                if mode == 'rgb_array':
                        return rgb_array
                else:
                        return rgb_array, mask

        def render_image(self,width, height, view_matrix, proj_matrix, shadow=1):
                (_, _, px, _, mask) = self._p.getCameraImage(width=width,
                                                        height=height,
                                                        viewMatrix=view_matrix,
                                                        projectionMatrix=proj_matrix,
                                                        shadow=shadow,
                                                        lightDirection=(10, 0, 10),
                                                        renderer=self._p.ER_BULLET_HARDWARE_OPENGL)

                rgb_array = np.array(px, dtype=np.uint8)
                rgb_array = np.reshape(rgb_array, (height, width, 4))

                rgb_array = rgb_array[:, :, :3]
                return rgb_array, mask

        def _render_callback(self, mode):
                pass
        
        # @property
        # def _max_episode_steps(self):
        #       return 26  #执行完正好26步 不懂要设置多少
   






# ##############原始的环境测试代码
# import wandb
# import random
# from omegaconf import DictConfig, OmegaConf
# import hydra

# @hydra.main(version_base=None, config_path="wandb_test_conf", config_name="config")     
# def wandb_init(cfg : DictConfig) -> None:
#     # wandb启动
    
#     wandb.init(
#         # set the wandb project where this run will be logged
#         project=cfg.project,

#         # track hyperparameters and run metadata
#         config=dict(cfg.config)
#     )
     
def main():
        #wandb_init()

        env = PandaGraspEnv(render_mode="human")  # create one process and corresponding env
        while True:
                env.test()

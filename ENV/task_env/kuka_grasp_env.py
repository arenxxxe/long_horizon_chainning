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
from .kuka_arm import Kuka
import random
import pybullet_data
from pkg_resources import parse_version

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960



from ..from_surrol.base_env.surrol_goalenv import SurRoLGoalEnv
from ..from_surrol.utils.pybullet_utils import reset_camera
from ..from_surrol.utils.pybullet_utils import get_link_pose, wrap_angle
#################################尝试直接改变环境代码################################
pybullet_wierd_offset=0.02
class KukaGraspEnv(SurRoLGoalEnv):
        #surol自带的3d模型
        ASSET_DIR_PATH="/home/wyq/SW/long_horizon_chainning/SIM_env/sim_env/from_surrol/3d_model/from_surrol"
        #调整各种位置和大小的东西 其实没啥用处
        ee_offset=0.25
        SCALING = 1.
        #A 完成上层调用逻辑     env = KukaGraspEnv() 
        #TODO 已经完成书写
        def __init__(self,render_mode='rgb_array'):
                
                #1设置渲染模式 开启bullet服务器--无需修改
                #2 相机参数设置-无需修改
                #3 种子设置--源代码未进行真正实现
                #4 参数配置-无需修改
                #5 物理仿真参数配置 加载地板 objid字典创建 --无需修改
                #6 场景设置--self._env_setup
                #7 初始goal拿到 -- self._sample_goal  
                #8 示教策略和subgoal设置--self._sample_goal_callback()
                #9 观察和动作空间设置-- _get_obs()   
                #10 仿真时间间隙-- 无需修正
                #man what can i say?  初始化的时候也是要拿obs 来做观察空间参数的初始化的
                self.touch_object_once=False
                super().__init__(render_mode=render_mode)

        @property
        def action_size(self):
                return 5 #xyz 末端执行器关节旋转 +夹爪开合
 
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
                if self._render_mode == 'human':
                        reset_camera(yaw=90.0, pitch=-30.0, dist=2 * self.SCALING,
                                        target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))

                #2 部署机器人和场景物体
                #2a 封装好的kuka机器人
                self._urdfRoot=pybullet_data.getDataPath()
                self._timeStep = 1. / 240.
                self._kuka = Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
                #2b 桌子
                p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -0.63000,
                        0.000000, 0.000000, 0.0, 1.0)
                #2c  随机扔几个物块
                xpos = 0.55 + 0.12 * 0.5
                ypos = 0 + 0.2 * 0.8
                ang = 3.14 * 0.5 + 3.1415925438 * 0.5
                orn = p.getQuaternionFromEuler([0, 0, ang])

                self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), xpos, ypos, 0.014,
                                        orn[0], orn[1], orn[2], orn[3])
                self.obj_ids['rigid'].append(self.blockUid)

                #3 导入显式路点的那个小红点模型
                obj_id = p.loadURDF(os.path.join(self.ASSET_DIR_PATH, 'sphere/sphere.urdf'),
                                globalScaling=self.SCALING*10)
                self.obj_ids['fixed'].append(obj_id)  # 0
                #简单测试区域 前面的东西认为是没问题的
        
                self.goal=self._sample_goal()
                self._sample_goal_callback()
                #self.kuka_action_control_test()
                        
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
                goal = [0.4039 , 0.5577 ,-0.991]
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
                red_point_init_pos=list(p.getBasePositionAndOrientation(self.obj_ids['rigid'][0])[0])

                p.resetBasePositionAndOrientation(self.obj_ids['fixed'][0], red_point_init_pos, (0, 0, 0, 1))
                
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
                
                

                init_object_pos=p.getBasePositionAndOrientation(self.obj_ids['rigid'][0])
                init_object_posy=init_object_pos[0][1]
                init_object_posx=init_object_pos[0][0]-0.02
                ee,joint_angle=self._kuka.getEE_pos()
                lift_pos=ee[2]
                
                open_close=[-0.035,0.3]
                                #1.1 先定位
                above_object_wp=[init_object_posx,init_object_posy,lift_pos,0,open_close[1]]
                                #1.2 接近物体
                reach_object_wp=[init_object_posx,init_object_posy,init_object_pos[0][2]+self.ee_offset,1.5,open_close[1]]                                
                                #1.3 抓取
                grasp_object_wp=[init_object_posx,init_object_posy,init_object_pos[0][2]+self.ee_offset,1.5,open_close[0]]                                
                
                                #1.4 抬升
                lift_object_wp=[init_object_posx,init_object_posy,lift_pos,1.5,open_close[0]]
                                #2.1 移动
                move_object_wp=[init_object_pos[0][0]-0.2,init_object_pos[0][1]+0.4,lift_pos,1.5,open_close[0]]
                                #2.2 释放
                release_object_wp=[init_object_pos[0][0]-0.2,init_object_pos[0][1]+0.4,lift_pos,1.5,open_close[1]]

                self._waypoints = [above_object_wp,reach_object_wp,grasp_object_wp,lift_object_wp,move_object_wp,release_object_wp] 
                #3 根据路点 设置子目标--物体移动的中间位置 动了才能叫作子目标 而不是末端执行器的 
                above_object_goal=[above_object_wp[0],above_object_wp[1],above_object_wp[2]]
                
                release_object_goal=[release_object_wp[0],release_object_wp[1],release_object_wp[2]]

                self.subgoals=[above_object_goal,release_object_goal]
                
        def _get_obs(self) :
                """
                处于的流程: A-9 场景设置  、B-3-5 传入action 完成一次仿真步进
                输入：无
                输出：dict:obs {'observation' 'achieved_goal' 'desired_goal'}
                目的: 拿到仿真中的机械臂和物体的实时信息
                TODO 需要进算法看一下输出是否符合算法要求格式
                np_array:robot_state 末端执行器 xyz + 三xyz欧拉角 + 末端开合角度
                np_array:observation 机器人状态robot_state 
                np_array:achieved_goal 物体xyz 
                np_array:desired_goal self.goal最终物体位置 
                dict:obs {'observation' 'achieved_goal' 'desired_goal'}
                
                """
                
                #1 机器人状态
                object_pos, _ = p.getBasePositionAndOrientation(self.blockUid)
                np_object_pos=np.array(object_pos)

                self.kuka_ee_state = self._kuka.getObservation()
                                #夹爪开合角度
                _,joint_angle=self._kuka.getEE_pos()
                
                #判断是否接近物体 接近很近了  就以物体作为达成目标 
                        #1 服用issuccess来判断物体是否接近

                np_ee_pos=np.array(self.kuka_ee_state[0:3])
                np_ee_pos[2]-=self.ee_offset#忘记了 末端执行器和夹爪之间有距离的
                #print(f"物体位置{np_object_pos}")
                #print(f"夹爪位置{np_ee_pos}")
                touch_once=self._is_success(np_ee_pos,np_object_pos,0.03)
                #print(f"互相接近的最短距离！！！！！！！！！{self.goal_distance(np_ee_pos, np_object_pos)}")         
                        #2 如果接近过一次 那之后 第三步一直必须成立
                if touch_once:
                        self.touch_object_once=True

                        #3 如果接近过一次 之后的达成目标就是物体位置
                if self.touch_object_once:
                        achieved_goal=np_object_pos

                self.kuka_ee_state.extend([joint_angle])  #4+1              
                
                robot_state=np.array(self.kuka_ee_state)
                        



                object_rel_pos1=object_pos - robot_state[0: 3]
                np_object_rel_pos1=np.array(object_rel_pos1)
                #参考的连接处状态
                wp_pos=[0.6119448196512544, 0.14788242760658257, 0.2006463000460834]
                wp_orn=list(p.getEulerFromQuaternion( (-0.00017743603891326752, -0.02160610316185721, 0.9994235856897068, -0.02618475109182154)))
                np_wp_pos=np.array(wp_pos)
                np_wp_orn=np.array(wp_orn)
                observation = np.concatenate([
                robot_state, np_object_pos.ravel(), np_object_rel_pos1.ravel(), np_wp_pos.ravel(),np_wp_orn.ravel()
                ])
                
                #2  已经达成目标 无非是物体位姿
                        #源代码核心逻辑：没有拿着物体自由移动的时候 没有达成的目标 不存在任何的达成目标 拿着物体的时候 物体位置就是达成目标
                        #TODO 实现判断物体是否被拿着的逻辑
                #achieved_goal = np.array(list(object_pos))
                #触发式的  就是没有达成某个条件之前 布尔值一直是false  达成一次之后 以后全部是true
                if not self.touch_object_once:
                        achieved_goal=np.array([0.000000000000000001,0.000000000000000001,0.000000000000000001])#0很特殊 可能导致维度不对

                #3 总观察
                obs = {
                'observation': observation.copy(),
                'achieved_goal': achieved_goal.copy(),
                'desired_goal': self.goal.copy()
                }

                return obs
        
        def kuka_action_control_test(self):
                #1 路点和目标处理为能执行的动作传递下去
                
                above=self._waypoints[0]
                above[2]+=pybullet_wierd_offset
                reach=self._waypoints[1]
                #怀疑0.25就是夹爪中心点和末端的距离了
                reach[2]+=pybullet_wierd_offset

                grasp=self._waypoints[2]
                grasp[2]+=pybullet_wierd_offset
                lift=self._waypoints[3]
                lift[2]+=pybullet_wierd_offset
                move=self._waypoints[4]
                move[2]+=pybullet_wierd_offset
                release=self._waypoints[5]
                release[2]+=pybullet_wierd_offset
                #import pdb;pdb.set_trace()
                #2 执行路点的操作
                ok = True
                for i in range(1):
                        #执行above
                        #可视化显示路点
                                time.sleep(5)
                                p.resetBasePositionAndOrientation(self.obj_ids['fixed'][0], [above[0],above[1],above[2]-0.25], (0, 0, 0, 1))
                        #执行above
                                for i in range(500):
                                        obs=self._get_obs()
                                        init_object_pos=p.getBasePositionAndOrientation(self.obj_ids['rigid'][0])
                                        self._kuka.applyAction(above)
                                        p.stepSimulation()
                                        time.sleep(1.0 / 30.0)

                        #执行reach
                                p.resetBasePositionAndOrientation(self.obj_ids['fixed'][0], [reach[0],reach[1],reach[2]-0.25], (0, 0, 0, 1))
                                for i in range(500):
                                        self._kuka.applyAction(reach)
                                        p.stepSimulation()
                                        time.sleep(1.0 / 30.0)
                        #执行抓取
                                p.resetBasePositionAndOrientation(self.obj_ids['fixed'][0], [grasp[0],grasp[1],grasp[2]-0.25], (0, 0, 0, 1))
                                for i in range(500):

                                        print("正在抓取")
                                        self._kuka.applyAction(grasp)
                                        p.stepSimulation()
                                        time.sleep(1.0 / 30.0)
                        #执行抬升
                                p.resetBasePositionAndOrientation(self.obj_ids['fixed'][0], [ lift[0], lift[1], lift[2]-0.25], (0, 0, 0, 1))
                                for i in range(500):
                                        self._kuka.applyAction( lift)
                                        p.stepSimulation()
                                        time.sleep(1.0 / 30.0)   
                                # for i in range(500):
                                #         print(f"物体位置旋转{p.getBasePositionAndOrientation(self.blockUid)}  ",flush=True,end=' ')

                                #         p.stepSimulation()
                                #         time.sleep(1.0 / 30.0)           
                        #执行移动
                                p.resetBasePositionAndOrientation(self.obj_ids['fixed'][0], [ move[0], move[1], move[2]-0.25], (0, 0, 0, 1))
                                for i in range(500):
                                        self._kuka.applyAction( move)
                                        p.stepSimulation()
                                        time.sleep(1.0 / 30.0)
                        #释放夹爪
                                p.resetBasePositionAndOrientation(self.obj_ids['fixed'][0], [ self.goal[0], self.goal[1], self.goal[2]], (0, 0, 0, 1))
                                for i in range(500):
                                        self._kuka.applyAction( release)
                                        p.stepSimulation()
                                        time.sleep(1.0 / 30.0)    
                                        #print(f"最终的位置：{p.getBasePositionAndOrientation(self.obj_ids['rigid'][0])} ",flush=True,end=" ")                                    

        #def _meet_contact_constraint_requirement(self) -> bool:
                #拿到物体的位置
                # pose = get_link_pose(self.obj_id, -1)
                # #高于目标的z轴的都认为需要加这个限制 
                # return pose[0][2] > self.goal[2] + 0.01 * self.SCALING 
        
        #def constrain_contact(self) -> bool:
                # self._contact_constraint=None
                # if self._contact_constraint is None:

                # # the grippers activate; to check if they can grasp the object
                # # TODO: check whether the constraint may cause side effects
                # psm = self.psm1 if self._activated == 0 else self.psm2
                # if self._meet_contact_constraint_requirement():
                # body_pose = p.getLinkState(psm.body, psm.EEF_LINK_INDEX)
                # obj_pose = p.getBasePositionAndOrientation(self.obj_id)
                # world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                # obj_to_body = p.multiplyTransforms(world_to_body[0],
                #                                         world_to_body[1],
                #                                         obj_pose[0], obj_pose[1])

                # self._contact_constraint = p.createConstraint(
                #         parentBodyUniqueId=psm.body,
                #         parentLinkIndex=psm.EEF_LINK_INDEX,
                #         childBodyUniqueId=self.obj_id,
                #         childLinkIndex=-1,
                #         jointType=p.JOINT_FIXED,
                #         jointAxis=(0, 0, 0),
                #         parentFramePosition=obj_to_body[0],
                #         parentFrameOrientation=obj_to_body[1],
                #         childFramePosition=(0, 0, 0),
                #         childFrameOrientation=(0, 0, 0))
                # # TODO: check the maxForce; very subtle
                # p.changeConstraint(self._contact_constraint, maxForce=20)        
                # pass
                
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

        def _set_action(self, action):
                """
                处于的流程: B-3-2 机器人设置动作
                输入： np_array:acton
                输出：  
                目的: 进行一次完整的仿真步进成功执行步骤
                TODO  
                """
                   
                list_action=action.tolist() 

               

                assert len(action) == self.action_size
                #如果是示教动作  全是0的时候说明示教路点已经走完 不应该允许再往下下发动作指令并仿真

                
                #一次动60  是源代码的一次动作的仿真步
                p.resetBasePositionAndOrientation(self.obj_ids['fixed'][0], [list_action[0],list_action[1],list_action[2]-self.ee_offset], (0, 0, 0, 1))
                list_action[2]+=0.02#因为pybullet奇怪的逆运动学计算 差为0.02 不确定是kuka机械臂的问题还是所有机械臂共性问题
                for i in range(60):
                        self._kuka.applyAction(list_action)
                        p.stepSimulation()
                        time.sleep(1.0 / 30.0)
           
        def _step_callback(self):
                #源代码中 用来进行模拟力封闭抓取的 思路就是使用pybullet的约束 直接把物体锁死在末端执行器上面 实现稳定抓取
                pass
           
        def goal_distance(self,goal_a, goal_b):
                assert goal_a.shape == goal_b.shape
                return np.linalg.norm(goal_a - goal_b, axis=-1)
        
        def _is_success(self, achieved_goal, desired_goal,threshold=0.005):
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
                #print(f"重置之后的末端执行器观察{obs['observation'][0:3]}")
                #重置之后 一定没有拿过 
                self.touch_object_once=False
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
                action=[0]*5
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
                        delta_yaw= (waypoint[3] - obs['observation'][3]).clip(-1, 1)
                        if np.abs(delta_pos).max() > 1:
                                delta_pos /= np.abs(delta_pos).max()
                        #print(f"位置原始差：{delta_pos}")
                        #print(f"欧拉角：{obs['observation'][3]},{obs['observation'][4]},{obs['observation'][5]}")
                        #print(f"路点旋转指令：{waypoint[3]}")
                        #print(f"旋转原始差：{delta_yaw}")
                        scale_factor = 0.4
                        delta_pos *= scale_factor 
                        #print(f"位置差{np.linalg.norm(delta_pos) * 0.01 / scale_factor}")
                        #print(f"旋转差{np.abs(delta_yaw)}  参考：{np.deg2rad(2.)}")
                        #4 判断是否到达位置 删路点
                        if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 2e-3 and np.abs(delta_yaw) < np.deg2rad(2.):
                                print(f"第{i+1}个路点已经执行完毕")
                                if i == 5:
                                        print("最后一个路点不会归零")

                                
                                self._waypoints[i] = None if i< len(self._waypoints)-1 else self._waypoints[i]
                        #5 没到达位置 继续执行原来路点
                        break
                return np.array(action),i
        
        #B 完成上层调用逻辑     env.test()    
        def test(self, horizon=250):
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

                        tic = time.time()
                        action,i = self.get_oracle_action(obs)
                        

                        print('\n -> step: {}, action: {}'.format(steps, np.round(action, 4)))
                        
                        obs, reward, done, info = self.step(action)
                        
                        print(f"奖励{reward}")
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
        #c 完成上层调用逻辑     env.render()
        def render(self, mode='rgb_array'):
                return super().render()

        def _render_callback(self, mode):
                pass
        
        # @property
        # def _max_episode_steps(self):
        #       return 26  #执行完正好26步 不懂要设置多少
   






##############原始的环境测试代码
if __name__ == "__main__":
        env = KukaGraspEnv()  # create one process and corresponding env
        env.test()
        rgb=env.render()
        import pdb;pdb.set_trace()


#     env.close()
#     time.sleep(2)

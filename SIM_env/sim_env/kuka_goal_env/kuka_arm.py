#bullet的kuka机械臂 去除工作空间限制
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data


class Kuka:

  def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01):
    self.urdfRootPath = urdfRootPath
    self.timeStep = timeStep
    self.maxVelocity = .35
    self.maxForce = 200.
    self.fingerAForce = 2
    self.fingerBForce = 2.5
    self.fingerTipForce = 2
    self.useInverseKinematics = 1
    self.useSimulation = 1
    self.useNullSpace = 21
    self.useOrientation = 1
    self.kukaEndEffectorIndex = 6
    self.kukaGripperIndex = 7
    #lower limits for null space
    self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
    #upper limits for null space
    self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
    #joint ranges for null space
    self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
    #restposes for null space
    self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    #joint damping coefficents
    self.jd = [
        0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
        0.00001, 0.00001, 0.00001, 0.00001
    ]
    self.reset()

  def reset(self):
    objects = p.loadSDF(os.path.join(self.urdfRootPath, "kuka_iiwa/kuka_with_gripper2.sdf"))
    self.kukaUid = objects[0]
    #for i in range (p.getNumJoints(self.kukaUid)):
    #  print(p.getJointInfo(self.kukaUid,i))
    p.resetBasePositionAndOrientation(self.kukaUid, [-0.100000, 0.000000, 0.070000],
                                      [0.000000, 0.000000, 0.000000, 1.000000])
    self.jointPositions = [
        0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
        -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
    ]
    self.numJoints = p.getNumJoints(self.kukaUid)
    for jointIndex in range(self.numJoints):
      p.resetJointState(self.kukaUid, jointIndex, self.jointPositions[jointIndex])
      p.setJointMotorControl2(self.kukaUid,
                              jointIndex,
                              p.POSITION_CONTROL,
                              targetPosition=self.jointPositions[jointIndex],
                              force=self.maxForce)

    self.trayUid = p.loadURDF(os.path.join(self.urdfRootPath, "tray/tray.urdf"), 0.640000,
                              0.075000, 0.0000, 0.000000, 0.000000, 1.000000, 0.000000)
    #self.endEffectorPos = [0.537, 0.0, 0.5]
    state = p.getLinkState(self.kukaUid, self.kukaEndEffectorIndex)
    self.endEffectorPos = list(state[0])
    self.endEffectorAngle = 0

    self.motorNames = []
    self.motorIndices = []
    
    for i in range(self.numJoints):
      jointInfo = p.getJointInfo(self.kukaUid, i)
      qIndex = jointInfo[3]
      if qIndex > -1:
        #print("motorname")
        #print(jointInfo[1])
        self.motorNames.append(str(jointInfo[1]))
        self.motorIndices.append(i)

  def getActionDimension(self):
    if (self.useInverseKinematics):
      return len(self.motorIndices)
    return 6  #position x,y,z and roll/pitch/yaw euler angles of end effector

  def getObservationDimension(self):
    return len(self.getObservation())

  def getObservation(self):
    observation = []
    state = p.getLinkState(self.kukaUid, self.kukaEndEffectorIndex)
    pos = state[0]
    orn = state[1]
    euler = p.getEulerFromQuaternion(orn)

    observation.extend(list(pos))
    observation.extend(list(euler))

    return observation


  #拿末端执行器位置
  def getEE_pos(self):
      state = p.getLinkState(self.kukaUid, self.kukaEndEffectorIndex)
      actualEndEffectorPos = state[0]
      
      ee_joint_postion_right=(p.getJointState(self.kukaUid,11))[0]
      ee_joint_postion_left=(p.getJointState(self.kukaUid,8))[0]
      ee_joint_open_angle=ee_joint_postion_right-ee_joint_postion_left
      return actualEndEffectorPos,ee_joint_open_angle
  
  def applyAction(self, target_ee):

    #1 逆运动学控制模式 只需要输入末端执行器期望的位姿
    if (self.useInverseKinematics):
      #目标位置转化为微分的目标 坐标相减得到相对量 目标坐标比原坐标大 减出来正数 原坐标+这个正数*微分量
      dv=1
      #写死的endeffector位置有问题 实时获取吧

      import numpy as np
      np_target_ee=np.array([target_ee[0],target_ee[1],target_ee[2],target_ee[3]])
      # self.endEffectorPosOrn=self.getObservation()
      # endEffectorPos=self.endEffectorPosOrn[0:3]
      # endEffectorPos=endEffectorPos.copy()
      
      endEffectorPos=self.endEffectorPos.copy()
      endEffectorPos.extend([self.endEffectorAngle])
      if not hasattr(self, "has_run"):
        print("这段代码仅执行一次")
        
        endEffectorPos[2]-=0.4
        self.endEffectorPos[2]-=0.4
        self.has_run = True
      
      np_endEffectorPosOrn=np.array( endEffectorPos)
      np_relative_posorn=np_target_ee-np_endEffectorPosOrn
         
      np_motorCommands=np_relative_posorn*dv
      motorCommands=np_motorCommands.tolist()

      #print(f"targetee{target_ee}")
      eeobs,_=self.getEE_pos()
      print(f"eeobs{eeobs[2]}")
      #print(self.endEffectorPos[2])
      #print(self.endEffectorPos[2])

      dx = motorCommands[0]
      dy = motorCommands[1]
      #关键认知：dz是来增内部的这个self.endEffectorPos 增一点 变化一点 和外界的真实的东西完全的脱离耦合的 dz越少 代表着什么？ 代表着主观上认为 一次是往 末端执行器目标位姿方向变化一点点
      #需要拿到关键数据   我给入某个dz下去 实际上的z方向上面的变化是多少？
      dz = motorCommands[2] *0.05
      da = motorCommands[3]
      fingerAngle = target_ee[4]
      

      #print("pos[2] (getLinkState(kukaEndEffectorIndex)")
      #print(actualEndEffectorPos[2])

      if abs(dx)<0.0001:
        dx=0
      
      self.endEffectorPos[0] = self.endEffectorPos[0] + dx
      
      #工作空间限制
      # if (self.endEffectorPos[0] > 0.56):
      #   self.endEffectorPos[0] = 0.56
      # if (self.endEffectorPos[0] < 0.50):
      #   self.endEffectorPos[0] = 0.50

      if abs(dy)<0.0001:
        dy=0
      self.endEffectorPos[1] = self.endEffectorPos[1] + dy
      #工作空间限制
      # if (self.endEffectorPos[1] < -0.17):
      #   self.endEffectorPos[1] = -0.17
      # if (self.endEffectorPos[1] > 0.1):
      #   self.endEffectorPos[1] = 0.1
      
      #print ("self.endEffectorPos[2]")
      #print (self.endEffectorPos[2])
      #print("actualEndEffectorPos[2]")
      #print(actualEndEffectorPos[2])
      #if (dz<0 or actualEndEffectorPos[2]<0.5):
      dz=dz
      
      if abs(dz)<0.1e-10:
        dz=0
      # print(dz)
      self.endEffectorPos[2] = self.endEffectorPos[2] + dz
      
      #print(f"位置{self.endEffectorPos[2]} dz{dz}")
      #工作空间限制
      # if (self.endEffectorPos[2] > 1):
      #   self.endEffectorPos[2] = 1
      # if (self.endEffectorPos[2] < -2):
      #   self.endEffectorPos[2] = -2
      #print(self.endEffectorPos[2])
  

      self.endEffectorAngle = self.endEffectorAngle + da

      pos = self.endEffectorPos
      #print( pos )
      orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
      
      if (self.useNullSpace == 1):
        if (self.useOrientation == 1):
          jointPoses = p.calculateInverseKinematics(self.kukaUid, self.kukaEndEffectorIndex, pos,
                                                    orn, self.ll, self.ul, self.jr, self.rp)
        else:
          jointPoses = p.calculateInverseKinematics(self.kukaUid,
                                                    self.kukaEndEffectorIndex,
                                                    pos,
                                                    lowerLimits=self.ll,
                                                    upperLimits=self.ul,
                                                    jointRanges=self.jr,
                                                    restPoses=self.rp)
      #走的是下面这里
      else:
        if (self.useOrientation == 1):
          jointPoses = p.calculateInverseKinematics(self.kukaUid,
                                                    self.kukaEndEffectorIndex,
                                                    pos,
                                                    orn,
                                                    jointDamping=self.jd)
        else:
          jointPoses = p.calculateInverseKinematics(self.kukaUid, self.kukaEndEffectorIndex, pos)

      #print("jointPoses")
      #print(jointPoses)
      #print("self.kukaEndEffectorIndex")
      #print(self.kukaEndEffectorIndex)
      if (self.useSimulation):
        for i in range(self.kukaEndEffectorIndex + 1):
          #print(i)
          p.setJointMotorControl2(bodyUniqueId=self.kukaUid,
                                  jointIndex=i,
                                  controlMode=p.POSITION_CONTROL,
                                  targetPosition=jointPoses[i],
                                  targetVelocity=0,
                                  force=self.maxForce,
                                  maxVelocity=self.maxVelocity,
                                  positionGain=0.3,
                                  velocityGain=1)
      
      else:
        #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
        for i in range(self.numJoints):
          p.resetJointState(self.kukaUid, i, jointPoses[i])
      #控制夹爪
      #1控制旋转
      p.setJointMotorControl2(self.kukaUid,
                              7,
                              p.POSITION_CONTROL,
                              targetPosition=self.endEffectorAngle,
                              force=self.maxForce)
      #2 控制夹爪开闭合
      p.setJointMotorControl2(self.kukaUid,
                              8,
                              p.POSITION_CONTROL,
                              targetPosition=-fingerAngle,
                              force=self.fingerAForce)
      p.setJointMotorControl2(self.kukaUid,
                              11,
                              p.POSITION_CONTROL,
                              targetPosition=fingerAngle,
                              force=self.fingerBForce)

      p.setJointMotorControl2(self.kukaUid,
                              10,
                              p.POSITION_CONTROL,
                              targetPosition=0,
                              force=self.fingerTipForce)
      
      p.setJointMotorControl2(self.kukaUid,
                              13,
                              p.POSITION_CONTROL,
                              targetPosition=0,
                              force=self.fingerTipForce)
    #不需要： 一个一个关节控制
    else:
      for action in range(len(motorCommands)):
        motor = self.motorIndices[action]
        p.setJointMotorControl2(self.kukaUid,
                                motor,
                                p.POSITION_CONTROL,
                                targetPosition=motorCommands[action],
                                force=self.maxForce)

from from_surrol.robots_wrapper.arm import Arm

import os
import numpy as np
import pybullet as p

##########################################################下面两个函数的辅助

def get_body_pose(body):
    raw = p.getBasePositionAndOrientation(body)
    position = list(raw[0])
    orn = list(raw[1])
    return [position, orn]
from collections import namedtuple
LinkState = namedtuple('LinkState', ['linkWorldPosition', 'linkWorldOrientation',
                                     'localInertialFramePosition', 'localInertialFrameOrientation',
                                     'worldLinkFramePosition', 'worldLinkFrameOrientation'])


def get_link_state(body, link):
    return LinkState(*p.getLinkState(body, link))

###########################################
JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                       'jointReactionForces', 'appliedJointMotorTorque'])


def get_joint_state(body, joint):
    return JointState(*p.getJointState(body, joint))


def get_joint_position(body, joint):
    return get_joint_state(body, joint).jointPosition


#############################################################最主要的两个函数
def get_link_pose(body, link):
    if link == BASE_LINK:
        return get_body_pose(body)
    # if set to 1 (or True), the Cartesian world position/orientation will be recomputed using forward kinematics.
    link_state = get_link_state(body, link)
    return [list(link_state.worldLinkFramePosition), list(link_state.worldLinkFrameOrientation)]

def get_joint_positions(body, joints=None):
    return list(get_joint_position(body, joint) for joint in joints)

######################################################################

#各个关节 负一是pybullet默认给第一关节的
LINKS = (
    "lbr_iiwa_link_0", "lbr_iiwa_link_1", "lbr_iiwa_link_2",  # -1, 0, 1 
    "lbr_iiwa_link_3", "lbr_iiwa_link_4",  # 2, 3
    "lbr_iiwa_link_5",  # 4
    "lbr_iiwa_link_6",  # 5
   "lbr_iiwa_link_7",  # 6
)

#TODO 不知道这东西怎么设置 关节运动限制 先不给限制吧
# 关节可能有运动范围限制
TOOL_JOINT_LIMIT = {
    'lower': np.deg2rad([-90.0, -45.0,   0.0, -np.inf]),  # not sure about the last joint
    'upper': np.deg2rad([ 90.0,  66.4, 254.0,  np.inf]),
}
TOOL_JOINT_LIMIT['lower'][2] = -0.01  # allow small tolerance
TOOL_JOINT_LIMIT['upper'][2] = 0.254  # prismatic joint (m); not sure, from ambf



#TODO 深入探查这些相机等坐标系的变换怎么影响机器人的控制
class Kuka_arm(Arm):
        #机器人信息 会被之后的东西用 
        NAME = 'Kuka'
        URDF_PATH = "kuka_iiwa/model_vr_limits.urdf"
        GRIPPER_SDF_FILE="gripper/wsg50_one_motor_gripper_new_free_base.sdf"
        
        DoF = 8  # 可以从urdf查看 机械臂本体有7个旋转关节 然后末端执行器一般是给一个自由度 虽然实质上末端执行器也有8个关节
        #关节类型
        JOINT_TYPES = ('R', 'R', 'R','R','R','R','R','P')
        
        EEF_LINK_INDEX = 8   # EEF link index, one redundant joint for inverse kinematics
        TIP_LINK_INDEX = 9   # 估计是摄像头放的地方吧
        #这不知道是啥东西
        # RCM_LINK_INDEX = 10  # RCM link index
        # D-H parameters
        A     = np.array([0.0, 0.0, 0.0, 0.0])
        ALPHA = np.array([np.pi / 2, -np.pi / 2, np.pi / 2, 0.0])
        D     = np.array([0.0, 0.0, -0.3822, 0.3829])
        THETA = np.array([np.pi / 2, -np.pi / 2, 0.0, 0.0])

        def __init__(self, pos=(0., 0., 1.), orn=(0., 0., 0., 1.),
                        scaling=1.):
                super(Arm, self).__init__(self.URDF_PATH, pos, orn,
                                        TOOL_JOINT_LIMIT, tool_T_tip, scaling)
                
                
                # 摄像机相关参数 
                self.view_matrix = None
                self.proj_matrix = None
                self._homo_delta = np.zeros((2, 1))
                self._wz = 0

                # 为了摄像机进行计算 旋转矩阵啥的
                pos_eef, orn_eef = get_link_pose(self.body, self.EEF_LINK_INDEX)
                pos_cam, orn_cam = get_link_pose(self.body, self.TIP_LINK_INDEX)
                self._tip_offset = np.linalg.norm(np.array(pos_eef) - np.array(pos_cam))  # TODO
                wRe = np.array(p.getMatrixFromQuaternion(orn_eef)).reshape((3, 3))
                wRc = np.array(p.getMatrixFromQuaternion(orn_cam)).reshape((3, 3))
                self._wRc0 = wRc.copy()  # initial rotation matrix
                self._eRc = np.matmul(wRe.T, wRc)

                gripper_setup()


                def gripper_setup(self,):
                        #load gripper and setup
                        kuka_gripper_id = p.loadSDF("gripper/wsg50_one_motor_gripper_new_free_base.sdf")[0]
                        # attach gripper to kuka arm
                        kuka_cid = p.createConstraint(kuka_id, 6, kuka_gripper_id, 0, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.05], [0, 0, 0])
                        kuka_cid2 = p.createConstraint(kuka_gripper_id, 4, kuka_gripper_id, 6, jointType=p.JOINT_GEAR, jointAxis=[1,1,1], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
                        p.changeConstraint(kuka_cid2, gearRatio=-1, erp=0.5, relativePositionTarget=0, maxForce=100)
                                # reset kuka
                        jointPositions = [-0.000000, -0.000000, 0.000000, 1.570793, 0.000000, -1.036725, 0.000001]
                        for jointIndex in range(p.getNumJoints(kuka_id)):
                                p.resetJointState(kuka_id, jointIndex, jointPositions[jointIndex])
                                p.setJointMotorControl2(kuka_id, jointIndex, p.POSITION_CONTROL, jointPositions[jointIndex], 0)
                
                #估计是考虑一些额外的的约束 所以会把输入的某些部分和关节当前的位置做计算 得到各种冗余关节的位置
                def _get_joint_positions_all(self, abs_input):
                        pass


                #相机计算物体在相机坐标系的坐标
                def get_centroid_proj(self, pos) -> np.ndarray:
                        pass
                #渲染图象 并不重要
                def render_image(self, width=RENDER_WIDTH, height=RENDER_HEIGHT):
                        pass



        
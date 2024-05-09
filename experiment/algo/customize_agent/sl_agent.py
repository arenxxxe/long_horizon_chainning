

#神经网络
#经典工厂模式 
    
##来点头文件 测试用
import torch.nn as nn
import torch
import  numpy as np
AGENTS = {
    'SL_DDPGBC': SkillLearningDDPGBC,
    'SL_DEX': SkillLearningDEX,
    'SC_DDPG': SkillChainingDDPG,
    'SC_AWAC': SkillChainingAWAC,
    'SC_SAC': SkillChainingSAC,
    'SC_SAC_SIL': SkillChainingSACSIL,
}




from gym.envs.registration import register


#尝试使用gym注册环境
#入口的接口不好说对不对 
#源代码的人 安装了唯一一个包 叫做surrol 就是一个文件夹 然后包的名字是surrol
# 所以它的entry point是     entry_point='surrol.tasks.needle_reach:NeedleReach',

register(
    id='KukaGrasp-v0',
    entry_point='chaining_package.ENV.task_env.kuka_grasp_env:KukaGraspEnv',
    max_episode_steps=120,
)



register(
    id='PandaGrasp-v0',
    entry_point='chaining_package.ENV.task_env.panda_grasp_env:PandaGraspEnv',
    max_episode_steps=60,
)
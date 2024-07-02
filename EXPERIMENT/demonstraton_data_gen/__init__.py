from gym.envs.registration import register


#尝试使用gym注册环境
#入口的接口不好说对不对 
#想想他是怎么运行的 这东西是怎么去找的
register(
    id='KukaGrasp-v0',
    entry_point='surrol.tasks.needle_reach:NeedleReach',
    max_episode_steps=50,
)
0

"""
Data generation for the case of Psm Envs and demonstrations.
Refer to
https://github.com/openai/baselines/blob/master/baselines/her/experiment/data_generation/fetch_data_generation.py
"""
import os
import argparse
import time
import numpy as np
import imageio
from chaining_package.ENV.will_be_deprecated.viskill_chaos_utility.const import ROOT_DIR_PATH
from chaining_package.ENV.agent_interface_env.panda_interface_env import PandaGraspSLWrapper
import gym

parser = argparse.ArgumentParser(description='generate demonstrations for imitation')
parser.add_argument('--env', type=str, required=False,default='PandaGrasp-v0',
                    help='the environment to generate demonstrations')
parser.add_argument('--video', action='store_true',
                    help='whether or not to record video')
parser.add_argument('--steps', type=int,
                    help='how many steps allowed to run')
parser.add_argument('--subtask', type=str,
                    help='how many steps allowed to run', default='grasp')
args = parser.parse_args()




global success_counter 

#开始的路点index-完成
SUBTASK_START = {
    'grasp': 0,
    'move':3
}

#下一个任务的开始路点  最终任务如果只有一个那就写结束的就行了 
#实际上目的是一件事：超出子任务的路点 你该怎么把他设置为0 让机械臂不动了
SUBTASK_END = {
    'grasp': 3,
    'move':5,
    'release': 5
}
actions = []
observations = []
infos = []
terminals = []
images = []  # record 0video
masks = []
gt_actions = []
def main():
    global actions, observations, infos, terminals, images, masks, gt_actions

    # 重置全局变量
    actions = []
    observations = []
    infos = []
    terminals = []
    images = []
    masks = []
    gt_actions = []
    #1 gym make的链路给我打通
    # train_env = gym.make(args.env, render_mode= 'rgb_array' ,state_save_id=0)  # 'human' 'rgb_array'

    # train_env = PandaGraspSLWrapper(train_env, output_raw_obs=True, subtask=args.subtask)
    eval_env = gym.make(args.env, render_mode= 'human',state_save_id=1)  # 'human' 'rgb_array'

    eval_env = PandaGraspSLWrapper(eval_env, output_raw_obs=True, subtask=args.subtask)
    # test(train_env)
        #1 gym make的链路给我打通
    # print("训练环境结束  开始测试环境..........................")
    # 重置全局变量
    actions = []
    observations = []
    infos = []
    terminals = []
    images = []
    masks = []
    gt_actions = []
    test(eval_env)





def test(env):

    #检查1 ：env与wrapper成功初始化
    #breakpoint() #1完成
    num_itr = 200 if not args.video else 10 #不用视频数据200次 要的话10次？
    cnt = 0
    success_counter = 0
    init_state_space = 'random'

    env.reset()
    #检查2 ：reset的返回值
    #breakpoint() #3 reset没看到明显的问题
    print("Reset!")
    init_time = time.time()
    # if env.env.env.state_save_id==1:
    #     breakpoint()
    if args.steps is None:
        args.steps = env.max_episode_steps #拿到的是子任务的

    #检查3 ：env.max_episode_steps属性
    #breakpoint() #没看到明显的问题
    print()
    while len(infos) < num_itr:
        #检查4 ：再次检查reset的返回值
        #print(f"循环停止条件{len(infos)}")
        obs_, obs = env.reset()
        print("ITERATION NUMBER ", len(infos))
        #5  检查能否满足goto的接口
        goToGoal(env, obs_, obs)
        # with open(f"{args.subtask}的奖励设置.txt","a") as file:
        #     file.write(f"第{len(infos) }个rollout记录完毕\n")
        cnt += 1
        #print(f"计数看看{cnt}")f                                                                                                                                       +][
               

    file_name = "data_"
    file_name += args.env
    file_name += "_" + init_state_space
    file_name += "_" + str(num_itr)
    file_name += "_primitive_new" + args.subtask + ".npz"
    folder = 'demo' if not args.video else 'video'
    #文件-文件夹-上层文件夹
    #breakpoint()
    current_file_path=os.path.abspath(__file__)
    current_dir_path=os.path.dirname(current_file_path)
    parent_dir_path=os.path.dirname(current_dir_path)

    storage_path=os.path.join(parent_dir_path,"data_storage")
    folder = os.path.join(storage_path, 'demonstration_data')
    # breakpoint()
    np.savez_compressed(os.path.join(folder, file_name),
                        actions=actions, observations=observations, terminals=terminals, gt_actions=gt_actions)  # save the file
    if args.video:
        video_name = "video"
        video_name += args.env + ".mp4"
        writer = imageio.get_writer(os.path.join(folder, video_name), fps=20)
        for img in images:
            writer.append_data(img)
        writer.close()

        if len(masks) > 0:
            mask_name = "mask_"
            mask_name += args.env + ".npz"
            np.savez_compressed(os.path.join(folder, mask_name),
                                masks=masks)  # save the file

    used_time = time.time() - init_time
    print("Saved data at:", folder)
    print("Time used: {:.1f}m, {:.1f}s\n".format(used_time // 60, used_time % 60))
    print(f"Trials: {num_itr}/{cnt}")
    #6 检查 关闭仿真环境的接口  
    env.close()


def goToGoal(env, last_obs_, last_obs):
    episode_acs = []
    episode_obs = []
    episode_info = []
    episode_terminals = []
    episode_gt_acs = []
    time_step = 0  # count the total number of time steps
    episode_init_time = time.time()
     
    episode_obs.append(last_obs_)

    obs_, obs, success = last_obs_, last_obs, False
    #执行一个episode 整个从子任务开始到结束

    while time_step < min(env.max_episode_steps, args.steps):
        #检查7 ：检查示教动作的接口
        action, i = env.get_oracle_action(obs)
        # action=np.round(action,4)

        #print(time_step)
        #准备到结束动作了 开始记录每次返回的施教动作
        # if i == SUBTASK_END[args.subtask]-1: 
        #     final_action=action
        #在结束动作处 按照上个施教动作的末尾动作运行
        #print(f"当前条件达成{i == SUBTASK_END[args.subtask]}\n")
        #逻辑上一定是 成功之后 不要动了
        # if 0:
        #     print(f"actions[4]:{action[4]}  \n")
        #     print(f"此时的奖励{reward}\n")
        #     print(f"此时的info{info['is_success']}\n")
        # print(f"此时的i{i}")
        if i == SUBTASK_END[args.subtask]:

            #最后一定要用0来停 不然数据有问题
            # info['is_success'] = 1
            
            if args.subtask == 'grasp':

                action = np.array([0,0,0,0],dtype=np.float32) 
                action[3]=-0.5

                #print(f"动作在{action}\n")

            elif args.subtask == 'move':

                action = np.array([0,0,0,0],dtype=np.float32) 
                action[3]=-0.5
                
                # print(action)
                    

            elif args.subtask == 'release':
                if success:
                    action = np.array([0,0,0,0],dtype=np.float32) 
                    action[3]=0.5
                
                #print(action)

            else:
                raise NotImplemented
            #print(f"最后的动作{action}当前的")
            #动作清零？ 目的是什么？代表着之后不要再拿新的下一个动作了 但是为什么是清零？ 都是给入绝对位姿
            #没有观察到归零之后向0走的迹象
            #检查最后数据里面有没有0的记录 有至少五个记录 说明这个东西也step下去了
            #作用猜测:断了之后的施教动作 待在原来结束的状态不变 他这里是通过其他set0和末端执行器原来状态达到的

 
        #print(f"时间步{time_step}")
        if args.video:
            #检查8 ：检查图片渲染接口
            img = env.render('rgb_array')
            images.append(img)
            # masks.append(mask)
        #检查8 ：检查环境步进接口 
        obs_, reward, done, info, obs = env.step(action)

        # with open(f"{args.subtask}的奖励设置.txt","a") as file:
        #     file.write(f"第{time_step}步长：奖励是{reward}\n")    
        # if i>=3:
        #     print(f" -> obs: {obs}, reward: {reward}, done: {done}, info: {info}.")
        time_step += 1
        #print(f"时间步{time_step}对应的动作id{i}")
        #print(reward, i)
        #print(f"成功的奖励信号{info['is_success']}")
                #避免最后掉落
        if success and reward== -1:
            success=False
            info['is_success']=0
            
        if isinstance(obs, dict) and info['is_success'] > 0 and not success:
            
            print("Timesteps to finish:", time_step)
            success = True



        # # if i >= 4 and i < 9:
        # if i >= SUBTASK_START[args.subtask] and i < SUBTASK_END[args.subtask]:
        episode_acs.append(action)
        episode_info.append(info)
        episode_obs.append(obs_)
        episode_terminals.append(done)
        episode_gt_acs.append(info['gt_goal'])
        last_obs_ = obs_
        
    print("Episode time used: {:.2f}s\n".format(time.time() - episode_init_time))
    print(f"成功没有？{success}")
    if success:

        actions.append(episode_acs)
        observations.append(episode_obs)
        infos.append(episode_info)
        terminals.append(episode_terminals)
        gt_actions.append(episode_gt_acs)


if __name__ == "__main__":
    main()

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
from chaining_package.ENV.agent_interface_env.kuka_slsc_wrapper import KukagraspSLWrapper
import gym

parser = argparse.ArgumentParser(description='generate demonstrations for imitation')
parser.add_argument('--env', type=str, required=False,default='KukaGrasp-v0',
                    help='the environment to generate demonstrations')
parser.add_argument('--video', action='store_true',
                    help='whether or not to record video')
parser.add_argument('--steps', type=int,
                    help='how many steps allowed to run')
parser.add_argument('--subtask', type=str,
                    help='how many steps allowed to run', default='grasp')
args = parser.parse_args()


actions = []
observations = []
infos = []
terminals = []
images = []  # record video
masks = []
gt_actions = []
global success_counter 

#开始的路点index-完成
SUBTASK_START = {
    'grasp': 0,
}

#结束的路点index-完成
SUBTASK_END = {
    'grasp': 4,
    'release': 6
}


def main():
    #1 gym make的链路给我打通
    env = gym.make(args.env, render_mode= 'rgb_array')  # 'human' 'rgb_array'

    env = KukagraspSLWrapper(env, output_raw_obs=True, subtask=args.subtask)
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
        cnt += 1
        #print(f"计数看看{cnt}")
        

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
    breakpoint()
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
        #print(time_step)
        #准备到结束动作了 开始记录每次返回的施教动作
        if i == SUBTASK_END[args.subtask]-1: 
            final_action=action
        #在结束动作处 按照上个施教动作的末尾动作运行

        if i >= SUBTASK_END[args.subtask]: 
            info['is_success'] = 1

            action = final_action #最后的动作一直保持
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
        # print(f" -> obs: {obs}, reward: {reward}, done: {done}, info: {info}.")
        time_step += 1
        #print(f"时间步{time_step}对应的动作id{i}")
        #print(reward, i)
        #print(f"成功的奖励信号{info['is_success']}")
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
    
    if success:

        actions.append(episode_acs)
        observations.append(episode_obs)
        infos.append(episode_info)
        terminals.append(episode_terminals)
        gt_actions.append(episode_gt_acs)


if __name__ == "__main__":
    main()

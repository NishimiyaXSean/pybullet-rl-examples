import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time
import numpy as np
import pybullet as p
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from marl_env import Drone1v1MARLEnv

RELATIVE_PATH = "./marl_checkpoints/run_0512_1429" 
CHECKPOINT_PATH = os.path.abspath(RELATIVE_PATH)

def env_creator(config):
    # 真正的 3D 渲染权，交给主线程里手动创建的 env
    return Drone1v1MARLEnv(gui=False)

if __name__ == "__main__":
    # 初始化 Ray
    ray.init()
    
    # 注册环境（必须与训练时一致）
    env_name = "drone_1v1_env"
    register_env(env_name, env_creator)

    # 从 Checkpoint 加载模型
    print(f"正在加载大脑记忆: {CHECKPOINT_PATH}")
    algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)
    
    # 实例化一个本地环境用于可视化
    env = Drone1v1MARLEnv(gui=True)
    
    print("==================================")
    print("1v1 多智能体对抗演习开始！")
    print("按键说明：[1-5] 切换运镜 | [ESC] 退出")
    print("==================================")

    # 循环进行多次演习测试
    for episode in range(10):
        obs, info = env.reset()
        terminated = {"__all__": False}
        truncated = {"__all__": False}
        
        # 记录每局的得分
        ep_reward_A = 0
        ep_reward_E = 0
        
        while not (terminated["__all__"] or truncated["__all__"]):
            # 双方大脑独立思考
            # 主机根据它的观测输出动作
            action_A = algo.compute_single_action(
                observation=obs["attacker_0"],
                policy_id="policy_attacker",
                explore=False # 测试时关闭随机探索
            )
            
            # 目标机根据它的观测输出动作
            action_E = algo.compute_single_action(
                observation=obs["evader_0"],
                policy_id="policy_evader",
                explore=False
            )
            
            actions = {
                "attacker_0": action_A,
                "evader_0": action_E
            }
            
            # 执行物理步进
            obs, rewards, terminated, truncated, infos = env.step(actions)
            
            # 累加奖励
            ep_reward_A += rewards.get("attacker_0", 0)
            ep_reward_E += rewards.get("evader_0", 0)
            
            # 保持 1:1 真实物理频率渲染
            time.sleep(0.02) 

            # 检测 ESC 退出
            keys = p.getKeyboardEvents()
            if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
                break
        
        print(f"第 {episode+1} 轮演习结束 | 主机得分: {ep_reward_A:.1f} | 目标机得分: {ep_reward_E:.1f}")
        
        if terminated.get("attacker_0") or terminated.get("evader_0"):
            print(">>> 战况结算：发生了有效的拦截或碰撞！")
        else:
            print(">>> 战况结算：演习超时，目标逃逸。")

    ray.shutdown()
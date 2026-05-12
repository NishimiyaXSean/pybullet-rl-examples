import os
# 解决 Windows 下 NumPy 和 PyTorch 的 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
import gymnasium as gym
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# 环境代码保存在 marl_env.py 中，类名叫 Drone1v1MARLEnv
from marl_env import Drone1v1MARLEnv

def env_creator(config):
    return Drone1v1MARLEnv(gui=False)

if __name__ == "__main__":
    # 1. 初始化 Ray 引擎
    ray.init()

    # 2. 注册环境名称
    env_name = "drone_1v1_env"
    register_env(env_name, env_creator)

    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(19,), dtype=np.float32)
    act_space = gym.spaces.Discrete(11)

    # 3. 核心算法配置 (PPOConfig)
    config = (
        PPOConfig()
        .environment(env=env_name)
        .framework("torch") # 必须指定使用 PyTorch
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
        .env_runners(num_env_runners=4) # 开启并行的 CPU 核心来跑环境收集数据
        
        # 强制关闭尚不成熟的新 API 栈
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        
        # 4. 多智能体策略分配 (Multi-Agent Setup)
        .multi_agent(
            # 定义两个独立的大脑
            policies={
                "policy_attacker": (None, obs_space, act_space, {}),
                "policy_evader": (None, obs_space, act_space, {}),
            },
            # 定义“谁”用“哪个大脑”的映射规则
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: 
                "policy_attacker" if agent_id == "attacker_0" else "policy_evader"
        )
        
        # 5. 神经网络结构 (Net Arch)
        .training(
            model={"fcnet_hiddens": [256, 256, 128], "fcnet_activation": "relu"},
            train_batch_size=4000,
            minibatch_size=256,
            lr=1e-4,
        )
    )

    # 6. 构建算法对象
    print("正在构建 RLlib 算法对象，请稍候...")
    algo = config.build()

    # 7. 开始训练循环
    TRAIN_ITERATIONS = 500
    CHECKPOINT_DIR = "./marl_checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("==================================")
    print("开始多智能体 1v1 空战对抗训练！")
    print("==================================")

    for i in range(TRAIN_ITERATIONS):
        # step() 会让所有 worker 跑环境，收集数据，更新神经网络，然后返回统计信息
        result = algo.train()
        
        # 打印双方的平均奖励，观察博弈胜负手
        reward_A = result['policy_reward_mean'].get('policy_attacker', 0.0)
        reward_E = result['policy_reward_mean'].get('policy_evader', 0.0)
        
        print(f"迭代 {i+1:03d} | "
              f"主机奖励: {reward_A:6.1f} | "
              f"目标机奖励: {reward_E:6.1f} | "
              f"总回合数: {result['episodes_total']}")

        # 每 50 次迭代保存一次模型
        if (i + 1) % 50 == 0:
            checkpoint_path = algo.save(CHECKPOINT_DIR)
            print(f"--> 模型已保存至: {checkpoint_path}")

    # 训练结束
    ray.shutdown()
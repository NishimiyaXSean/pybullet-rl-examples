import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

# 引入你的环境和模型
from marl_env import Drone1v1MARLEnv
from mappo_model import MAPPOModel

def env_creator(config):
    env = Drone1v1MARLEnv(gui=False)
    env.set_curriculum_stage(3) # 强制在最高难度 Stage 3 下进行公平测试
    return env

def get_historical_checkpoints(run_dir):
    """扫描目录，提取所有周期性保存的 Checkpoints 并按迭代次数排序"""
    checkpoint_dirs = glob.glob(os.path.join(run_dir, "checkpoints", "checkpoint_[0-9]*"))
    
    # 解析出迭代次数并排序
    ckpt_list = []
    for d in checkpoint_dirs:
        match = re.search(r'checkpoint_(\d+)', d)
        if match:
            iter_num = int(match.group(1))
            ckpt_list.append((iter_num, d))
            
    # 按迭代次数从小到大排序
    ckpt_list.sort(key=lambda x: x[0])
    return ckpt_list

def build_eval_algo():
    """构建用于纯推理的轻量级 RLlib 算法对象 (不启动并行 Worker 节省显存)"""
    config = (
        PPOConfig()
        .environment("drone_1v1_mappo_env")
        .framework("torch")
        .env_runners(num_env_runners=0) # 纯推理模式，不需要 rollout workers
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .multi_agent(
            policies={
                "policy_attacker": (None, obs_space, act_space, {}),
                "policy_evader":   (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: 
                "policy_attacker" if agent_id == "attacker_0" else "policy_evader",
        )
        .training(model={"custom_model": "mappo_centralized_critic"})
    )
    return config.build()

if __name__ == "__main__":
    # ================= 1. 配置路径 =================
    # 填入正在跑的，或者已经跑完的 MARL 训练文件夹路径
    TARGET_RUN_DIR = "./marl_runs/mappo_run_0606_0923/checkpoints/checkpoint_best_iter_507" 
    # ===============================================

    ray.init()
    register_env("drone_1v1_mappo_env", env_creator)
    ModelCatalog.register_custom_model("mappo_centralized_critic", MAPPOModel)

    # 获取空间维度
    temp_env = env_creator({})
    obs_space = temp_env.observation_spaces["attacker_0"]
    act_space = temp_env.action_spaces["attacker_0"]

    # 获取所有历史模型
    historical_ckpts = get_historical_checkpoints(TARGET_RUN_DIR)
    if not historical_ckpts:
        print("未找到历史 Checkpoints，请检查路径。")
        exit()

    latest_iter, latest_ckpt = historical_ckpts[-1]
    print(f"找到 {len(historical_ckpts)} 个历史模型。")
    print(f"当前最新主战模型为: Iteration {latest_iter}")

    # ================= 2. 双开算法实例 =================
    print("正在加载 Challenger (当前最新攻击机)...")
    algo_challenger = build_eval_algo()
    algo_challenger.restore(latest_ckpt)

    print("正在加载 Ghost (历史目标机挂载器)...")
    algo_ghost = build_eval_algo()

    # ================= 3. 开始跨代对抗测试 =================
    TEST_GAMES_PER_GHOST = 30 # 每个历史版本打 30 局
    eval_env = env_creator({})
    
    results_x_iters = []
    results_y_winrates = []

    for ghost_iter, ghost_ckpt in historical_ckpts:
        print(f"\n[{latest_iter} 的攻击机] VS [历史 {ghost_iter} 的目标机]")
        
        # 将幽灵模型加载到算法 2 中
        algo_ghost.restore(ghost_ckpt)
        
        wins = 0
        for game in range(TEST_GAMES_PER_GHOST):
            obs, info = eval_env.reset()
            terminated = {"__all__": False}
            truncated = {"__all__": False}
            final_reason = "timeout"
            
            while not (terminated["__all__"] or truncated["__all__"]):
                actions = {}
                # 攻击机的大脑来自于 algo_challenger
                if "attacker_0" in obs:
                    actions["attacker_0"] = algo_challenger.compute_single_action(
                        obs["attacker_0"], policy_id="policy_attacker", explore=False
                    )
                # 目标机的大脑来自于 algo_ghost (历史模型)
                if "evader_0" in obs:
                    actions["evader_0"] = algo_ghost.compute_single_action(
                        obs["evader_0"], policy_id="policy_evader", explore=False
                    )
                    
                obs, rewards, terminated, truncated, infos = eval_env.step(actions)
                
                if "attacker_0" in infos and "reason" in infos["attacker_0"]:
                    final_reason = infos["attacker_0"]["reason"]
            
            if final_reason == "success":
                wins += 1
                
        win_rate = wins / TEST_GAMES_PER_GHOST
        print(f"--> 实测结果: 击杀率 {win_rate*100:.1f}% ({wins}/{TEST_GAMES_PER_GHOST})")
        
        results_x_iters.append(ghost_iter)
        results_y_winrates.append(win_rate)

    # ================= 4. 绘制“真·战斗力”曲线 =================
    plt.figure(figsize=(10, 6))
    plt.plot(results_x_iters, results_y_winrates, marker='o', linestyle='-', color='b', linewidth=2)
    plt.fill_between(results_x_iters, 0, results_y_winrates, color='b', alpha=0.1)
    
    plt.title(f"True Combat Power: Challenger (Iter {latest_iter}) vs Historical Ghosts", fontsize=14)
    plt.xlabel("Historical Opponent Iteration (Ghost Age)", fontsize=12)
    plt.ylabel(f"Win Rate of Latest Attacker", fontsize=12)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    save_path = os.path.join(TARGET_RUN_DIR, "true_combat_power_curve.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n绘图完成！真·战斗力评估曲线已保存至: {save_path}")
    
    ray.shutdown()
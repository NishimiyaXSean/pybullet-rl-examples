import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from marl_env import Drone1v1MARLEnv

RELATIVE_PATH = "./marl_runs/run_0520_1603/checkpoints/checkpoint_final" 
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

    algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)
    print("模型加载完成！")

    # ==========================================
    # 模式切换开关：
    # "STRESS_TEST" -> 后台极速运行 100 局，统计真实胜率
    # "VISUAL_TEST" -> 带有 3D 界面和图表分析的单局观看模式
    # ==========================================
    RUN_MODE = "VISUAL_TEST" 

    if RUN_MODE == "STRESS_TEST":
        print("==================================")
        print("开启百场极速高压测试 (Headless Mode)...")
        print("==================================")
        
        # 压测必须关闭 GUI 以解锁最高运算速度
        test_env = Drone1v1MARLEnv(gui=False)
        
        TOTAL_EPISODES = 100
        stats = {"success": 0, "ground_crash": 0, "out_of_bounds": 0, "timeout": 0}

        for episode in range(TOTAL_EPISODES):
            obs, info = test_env.reset()
            terminated = {"__all__": False}
            truncated = {"__all__": False}
            
            final_reason = "timeout" # 默认结果

            while not (terminated["__all__"] or truncated["__all__"]):
                action_A = algo.compute_single_action(obs["attacker_0"], policy_id="policy_attacker", explore=False)
                action_E = algo.compute_single_action(obs["evader_0"], policy_id="policy_evader", explore=False)
                
                obs, rewards, terminated, truncated, infos = test_env.step({
                    "attacker_0": action_A,
                    "evader_0": action_E
                })
                
                # 实时提取并更新死因
                if "attacker_0" in infos and "reason" in infos["attacker_0"]:
                    final_reason = infos["attacker_0"]["reason"]

            # 统计结果
            if final_reason in stats:
                stats[final_reason] += 1
            else:
                stats["timeout"] += 1
                
            # 每打完 10 局播报一次进度
            if (episode + 1) % 10 == 0:
                print(f"压测进度: {episode + 1}/{TOTAL_EPISODES} 局完成...")

        print("\n" + "="*40)
        print("百场高强度压测最终战报")
        print("="*40)
        print(f"成功击杀 (Success) : {stats['success']} 局 ({stats['success']/TOTAL_EPISODES*100:.1f}%)")
        print(f"意外坠地 (Crash)   : {stats['ground_crash']} 局 ({stats['ground_crash']/TOTAL_EPISODES*100:.1f}%)")
        print(f"违规越界 (OOB)     : {stats['out_of_bounds']} 局 ({stats['out_of_bounds']/TOTAL_EPISODES*100:.1f}%)")
        print(f"演习超时 (Timeout) : {stats['timeout']} 局 ({stats['timeout']/TOTAL_EPISODES*100:.1f}%)")
        print("="*40)

    elif RUN_MODE == "VISUAL_TEST":

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

            # --- 绘图数据收集器 ---
            history_pos_A = []
            history_pos_E = []
            history_dist = []
            
            while not (terminated["__all__"] or truncated["__all__"]):
                # 双方大脑独立思考
                actions = {}

                # 主机根据它的观测输出动作
                if "attacker_0" in obs:
                    actions["attacker_0"] = algo.compute_single_action(
                        observation=obs["attacker_0"],
                        policy_id="policy_attacker",
                        explore=False # 测试时关闭随机探索
                    )
                if "evader_0" in obs:
                    actions["evader_0"] = algo.compute_single_action(
                        observation=obs["evader_0"],
                        policy_id="policy_evader",
                        explore=False
                    )
                
                # 执行物理步进    
                obs, rewards, terminated, truncated, infos = env.step(actions)      
                
                # 累加奖励
                ep_reward_A += rewards.get("attacker_0", 0)
                ep_reward_E += rewards.get("evader_0", 0)
                
                # --- 记录当前帧的绝对物理坐标 ---
                # 通过 env.pyb_env 绕过观测空间，直接获取上帝视角的真实位置
                pos_A = env.pyb_env._getDroneStateVector(0)[0:3]
                pos_E = env.pyb_env._getDroneStateVector(1)[0:3]
                dist = np.linalg.norm(pos_A - pos_E)
                
                history_pos_A.append(pos_A)
                history_pos_E.append(pos_E)
                history_dist.append(dist)

                # 检测 ESC 退出
                keys = p.getKeyboardEvents()
                if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
                    break
            
            print(f"第 {episode+1} 轮演习结束 | 主机得分: {ep_reward_A:.1f} | 目标机得分: {ep_reward_E:.1f}")
            
            # 改进这里的播报逻辑，利用环境传出的 reason
            reason = infos.get("attacker_0", {}).get("reason", "timeout")
            if reason == "success":
                print(">>> 战况结算：发生了有效的拦截或碰撞！")
            elif reason == "ground_crash":
                print(">>> 战况结算：主机意外坠地！")
            elif reason == "out_of_bounds":
                print(">>> 战况结算：主机违规越界！")
            else:
                print(">>> 战况结算：演习超时，目标逃逸。")

            # 演习结束，生成战报分析图表
            arr_pos_A = np.array(history_pos_A)
            arr_pos_E = np.array(history_pos_E)
            
            fig = plt.figure(figsize=(16, 6))
            plt.suptitle(f"Episode {episode+1} Combat Analysis", fontsize=16, fontweight='bold')

            # 图表 1：相对距离曲线
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.plot(history_dist, label='Relative Distance', color='blue', linewidth=2)
            ax1.axhline(y=150.0, color='orange', linestyle='--', label='Fuze Trigger Radius (150m)')
            ax1.axhline(y=50.0, color='red', linestyle='--', label='Kinetic Hit (50m)')
            ax1.set_title("Interception Distance over Time")
            ax1.set_xlabel("Time Steps")
            ax1.set_ylabel("Distance (m)")
            ax1.grid(True, alpha=0.5)
            ax1.legend()

            # 图表 2：3D 狗斗轨迹
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            ax2.plot(arr_pos_A[:, 0], arr_pos_A[:, 1], arr_pos_A[:, 2], label='Attacker', color='red', linewidth=2)
            ax2.plot(arr_pos_E[:, 0], arr_pos_E[:, 1], arr_pos_E[:, 2], label='Evader (Target)', color='orange', linewidth=2)
            
            # 标记起点和终点
            ax2.scatter(arr_pos_A[0, 0], arr_pos_A[0, 1], arr_pos_A[0, 2], color='darkred', marker='o', s=50, label='Start A')
            ax2.scatter(arr_pos_E[0, 0], arr_pos_E[0, 1], arr_pos_E[0, 2], color='goldenrod', marker='o', s=50, label='Start E')
            # 绘制相撞/终点标识
            if reason == "success":
                ax2.scatter(arr_pos_A[-1, 0], arr_pos_A[-1, 1], arr_pos_A[-1, 2], color='red', marker='*', s=300, label='KILL POINT')
            else:
                ax2.scatter(arr_pos_A[-1, 0], arr_pos_A[-1, 1], arr_pos_A[-1, 2], color='black', marker='x', s=100, label='End Point')

            ax2.set_title("3D Dogfight Trajectory")
            ax2.set_xlabel("X (m)")
            ax2.set_ylabel("Y (m)")
            ax2.set_zlabel("Z (m)")
            ax2.legend()

            plt.tight_layout()
            print("请查看弹出的 3D 轨迹分析图。关闭图表窗口后，将自动开始下一局演习。")
            plt.show()

    ray.shutdown()
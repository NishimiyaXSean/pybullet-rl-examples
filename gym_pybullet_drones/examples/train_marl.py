import os
import shutil  # 用于删除旧的最优模型文件夹
import datetime
current_time = datetime.datetime.now().strftime("%m%d_%H%M")
PROJECT_ROOT = os.path.abspath(f"./marl_runs/run_{current_time}")
os.environ['TUNE_RESULT_DIR'] = PROJECT_ROOT
os.environ['RAY_RESULTS'] = PROJECT_ROOT
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 解决 Windows 下 NumPy 和 PyTorch 的 OpenMP 冲突
os.environ['RAY_CHDIR_TO_TRIAL_DIR'] = '0' # 防止工作目录被意外篡改

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# 环境代码保存在 marl_env.py 中，类名叫 Drone1v1MARLEnv
from marl_env import Drone1v1MARLEnv

def env_creator(config):
    # 1. 实例化环境
    env = Drone1v1MARLEnv(gui=False)

    # 2. 从 RLlib 的配置字典中读取性能系数（如果存在的话）
    if "evader_speed_coeff" in config:
        env.EVADER_SPEED_COEFF = config["evader_speed_coeff"]
    if "evader_g_coeff" in config:
        env.EVADER_G_COEFF = config["evader_g_coeff"]
        
    return env

class DroneMetricsCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        # 尝试获取攻击机在最后一帧的 info 字典
        info = episode.last_info_for("attacker_0")

        if info:
            reason = info.get("reason", "timeout")
        else:
            reason = "timeout" # 如果没有 info，说明是时间耗尽平局
        
        '''
        # 精细化拆解指标 (True -> 1.0, False -> 0.0)
        # 1. 成功率: 真正进入有效射程
        episode.custom_metrics["rate_success"] = 1.0 if reason == "success" else 0.0
        
        # 2. 坠地率
        episode.custom_metrics["rate_crash"] = 1.0 if reason == "ground_crash" else 0.0
        
        # 3. 越界率
        episode.custom_metrics["rate_oob"] = 1.0 if reason == "out_of_bounds" else 0.0
        
        # 4. 超时率: 目标机成功存活到了回合结束
        episode.custom_metrics["rate_timeout"] = 1.0 if reason == "timeout" else 0.0

        '''

        episode.hist_data["rate_success"] = [1.0 if reason == "success" else 0.0]
        episode.hist_data["rate_crash"] = [1.0 if reason == "ground_crash" else 0.0]
        episode.hist_data["rate_oob"] = [1.0 if reason == "out_of_bounds" else 0.0]
        episode.hist_data["rate_timeout"] = [1.0 if reason == "timeout" else 0.0]

if __name__ == "__main__":
    # 1. 初始化 Ray 引擎
    ray.init()

    # 2. 注册环境名称
    env_name = "drone_1v1_env"
    register_env(env_name, env_creator)

    # 动态获取空间维度
    temp_env = env_creator({})
    obs_space = temp_env.observation_spaces["attacker_0"]
    act_space = temp_env.action_spaces["attacker_0"]
    print(f"检测到环境观测空间维度: {obs_space.shape}, 动作空间: {act_space.n}")

    # 3. 核心算法配置 (PPOConfig)
    config = (
        PPOConfig()
        .environment(env=env_name)
        .framework("torch") # 必须指定使用 PyTorch
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
        .env_runners(
            num_env_runners=4,
            sample_timeout_s=300,      # 将超时容忍度从默认的 60 秒延长到 5 分钟
            rollout_fragment_length=256 # 细化数据包，避免单次收集太久
            ) 
        .callbacks(DroneMetricsCallback)
        
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
                "policy_attacker" if agent_id == "attacker_0" else "policy_evader",

            # =============== 新增优化 ===============
            # 在 Phase 1 阶段，只训练攻击机的大脑，目标机大脑完全冻结不参与计算
            policies_to_train=["policy_attacker"]
            # ========================================
        )
        
        # 5. 神经网络结构 (Net Arch)
        .training(
            model={"fcnet_hiddens": [256, 256, 128], "fcnet_activation": "relu"},
            train_batch_size=8192,
            minibatch_size=1024,
            lr=3e-4,
            entropy_coeff=0.1,
            # 限制价值函数的截断 (Clip Param)
            clip_param=0.2,
            vf_clip_param=10.0,
        )
    )

    # 6. 构建算法对象
    print("正在构建 RLlib 算法对象，请稍候...")

    algo = config.build()

    # 创建独立的权重存放子文件夹
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("\n" + "="*45)
    print("TensorBoard 日志准备就绪！")
    print("请新开一个终端（Terminal），运行以下命令查看实时曲线：")
    print(f"tensorboard --logdir=\"{PROJECT_ROOT}\"")
    print("="*45 + "\n")

    # 加载旧模型以继续训练
    OLD_CHECKPOINT = os.path.abspath("./marl_runs/run_0521_1624/checkpoints/checkpoint_best_iter_492" )

    if os.path.exists(OLD_CHECKPOINT):
        print(f"正在恢复旧模型记忆: {OLD_CHECKPOINT}")
        algo.restore(OLD_CHECKPOINT)
    else:
        print("未发现旧模型，将从随机初始化开始全新训练。")
    

    tb_writer = SummaryWriter(log_dir=PROJECT_ROOT)

    # 7. 开始训练循环
    TRAIN_ITERATIONS = 500
    best_success_rate = -0.01   # 初始化成功率为 -0.01，这样可以确保第一轮训练（即使成功率是 0%）也能作为保底模型保存下来
    best_checkpoint_path = None    
    global_episodes = 0  # 新增：全局回合计数器 

    print("==================================")
    print("开始多智能体 1v1 空战对抗训练！")
    print("提示：在终端按下 【Ctrl + C】 可随时安全终止训练并保存模型！")
    print("==================================")

    try: 
        for i in range(TRAIN_ITERATIONS): # 每一次迭代为train_batch_size = 8192步
            # step() 会让所有 worker 跑环境，收集数据，更新神经网络，然后返回统计信息
            result = algo.train()

            # 尝试从 env_runners 中获取数据，如果没有则退回使用 result 本身
            stats = result.get("env_runners", result)
            policy_rewards = stats.get("policy_reward_mean", {})
            
            # 打印双方的平均奖励，观察博弈胜负手
            reward_A = policy_rewards.get("policy_attacker", 0.0)
            reward_E = policy_rewards.get("policy_evader", 0.0)

            # 提取总训练步数和本轮完成的回合数
            total_steps = result.get("num_env_steps_trained", 0)

            # 【修复点1】获取本轮准确的回合数 (RLlib 的标准键名是 episodes_this_iter)
            episodes_this_iter = stats.get("episodes_this_iter", 0)

            # ================= 核心修复：精准提取本轮迭代的真实统计 =================
            hist_stats = stats.get("hist_stats", {})

            # 提取历史记录列表 (RLlib 默认保留最近的 100 局)
            success_list = hist_stats.get("rate_success", [])
            crash_list   = hist_stats.get("rate_crash", [])
            oob_list     = hist_stats.get("rate_oob", [])
            timeout_list = hist_stats.get("rate_timeout", [])

            # 【修复点2】辅助函数：利用切片 (Slicing) 强制只取最后 N 局的数据
            def calc_iter_mean(lst, num_recent):
                if num_recent <= 0 or not lst:
                    return 0.0
                # 取列表最后的 num_recent 个元素
                recent_lst = lst[-num_recent:]
                return sum(recent_lst) / len(recent_lst)

            # 现在的率值，严格等于本轮这二十多局的真实表现！
            success_rate = calc_iter_mean(success_list, episodes_this_iter)
            crash_rate   = calc_iter_mean(crash_list, episodes_this_iter)
            oob_rate     = calc_iter_mean(oob_list, episodes_this_iter)
            timeout_rate = calc_iter_mean(timeout_list, episodes_this_iter)
            # =====================================================================

            # 提取策略熵 (Entropy) 
            learner_info = result.get("info", {}).get("learner", {})
            attacker_learner = learner_info.get("policy_attacker", {})
            
            # 兼容 RLlib 的不同嵌套层级
            learner_stats = attacker_learner.get("learner_stats", attacker_learner)
            
            # 提取 Entropy
            entropy = learner_stats.get("entropy", 0.0)
            
            print(f"迭代 {i+1:03d} | "
                  f"奖励(主/敌): {reward_A:6.1f} / {reward_E:6.1f} | "
                  f"本轮真实终局 -> 击杀:{success_rate*100:5.1f}% | 坠地:{crash_rate*100:5.1f}% | 越界:{oob_rate*100:5.1f}% | 超时:{timeout_rate*100:5.1f}% | "
                  f"本轮局数: {episodes_this_iter:3d} | "
                  f"熵: {entropy:.4f} | "
                  f"总训练步数: {total_steps}")
            
            # 写入宏观平均曲线 (横坐标为 Iteration)
            tb_writer.add_scalar("1_Rewards/Attacker", reward_A, i+1)
            tb_writer.add_scalar("1_Rewards/Evader", reward_E, i+1)
            
            tb_writer.add_scalar("2_Combat_Rates/Success_Kill", success_rate * 100, i+1)
            tb_writer.add_scalar("2_Combat_Rates/Ground_Crash", crash_rate * 100, i+1)
            tb_writer.add_scalar("2_Combat_Rates/Out_of_Bounds", oob_rate * 100, i+1)
            tb_writer.add_scalar("2_Combat_Rates/Timeout", timeout_rate * 100, i+1)
            tb_writer.add_scalar("5_Network_Stats/Entropy", entropy, i+1)
            
            # RLlib 默认会将 policy 奖励存为 "policy_{policy_id}_reward"
            a_rewards_hist = hist_stats.get("policy_policy_attacker_reward", [])
            e_rewards_hist = hist_stats.get("policy_policy_evader_reward", [])
            
            # 你在 callback 里记录的 custom_metrics 也会原封不动保存在这里
            success_hist = hist_stats.get("rate_success", [])

            # 遍历这一轮收集到的所有完整回合
            for idx in range(len(a_rewards_hist)):
                global_episodes += 1 # 推进全局回合数
                
                # 记录这一局两架飞机的真实得分
                tb_writer.add_scalar("3_Micro_Per_Episode/Attacker_Reward", a_rewards_hist[idx], global_episodes)
                
                if idx < len(e_rewards_hist):
                    tb_writer.add_scalar("3_Micro_Per_Episode/Evader_Reward", e_rewards_hist[idx], global_episodes)
                
                # 记录这一局是否发生了击杀 (1.0 代表成功，0.0 代表没成功)
                # 这在图表上会形成 0 和 1 的散点图，非常直观！
                if idx < len(success_hist):
                    tb_writer.add_scalar("4_Micro_Events/Is_Success", success_hist[idx], global_episodes)
            # ==============================================================
            
            tb_writer.flush() # 强制立刻写盘，绝不缓存延迟！
            
            # 保存最高成功率模型
            if success_rate > best_success_rate:
                # 针对 0% 的初次保存做个特殊打印，后面的正常打印提升比例
                if best_success_rate < 0:
                    print(f"建立初始战术基线！当前成功率：{success_rate * 100:.1f}%")
                else:
                    print(f"战术突破！发现新的最高成功率：{best_success_rate * 100:.1f}% -> {success_rate * 100:.1f}%")
                
                best_success_rate = success_rate
                
                # 构建带有迭代次数的新文件夹名称
                new_best_dir = os.path.join(CHECKPOINT_DIR, f"checkpoint_best_iter_{i+1:03d}")
                
                # 保存最新的最优模型
                algo.save(new_best_dir) 
                print(f"--> [最优] 模型已保存至: {new_best_dir}")
                
                # 如果之前已经有最优模型了，将其彻底删除
                if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                    shutil.rmtree(best_checkpoint_path, ignore_errors=True)
                
                # 更新指针，指向刚刚保存的这个新模型
                best_checkpoint_path = new_best_dir    

            # 每 50 次迭代保存一次模型
            if (i + 1) % 50 == 0:
                current_save_path = os.path.join(CHECKPOINT_DIR,f"checkpoint_{i+1:06d}")
                algo.save(current_save_path)
                print(f"--> 模型已保存至: {current_save_path}")

    except KeyboardInterrupt:
        # 当你按下 Ctrl+C 时，会跳到这里执行
        print("\n==================================")
        print("收到中止信号 (Ctrl+C)！正在执行安全退出并提取大脑记忆...")
        final_save_path = os.path.join(CHECKPOINT_DIR, "checkpoint_final")
        algo.save(final_save_path)
        print(f"--> [最终保存] 模型已安全暂存至: {final_save_path}")
        print("==================================")

    finally:
        # 无论正常跑完还是被中断，都确保关闭 Ray 引擎，释放内存
        ray.shutdown()
        tb_writer.close() # 关闭写入器
        print("训练脚本已安全关闭。")
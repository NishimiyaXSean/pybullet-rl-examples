import os
import shutil  # 用于删除旧的最优模型文件夹
import datetime
current_time = datetime.datetime.now().strftime("%m%d_%H%M")
PROJECT_ROOT = os.path.abspath(f"./marl_runs/mappo_run_{current_time}")
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

# 引入自定义 MAPPO 网络和模型注册器 
from ray.rllib.models import ModelCatalog
from mappo_model import MAPPOModel

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
    def __init__(self):
        super().__init__()
        self.current_stage = 1 # 初始化阶段

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        info = episode.last_info_for("attacker_0")
        reason = info.get("reason", "timeout") if info else "timeout"

        episode.hist_data["rate_success"] = [1.0 if reason == "success" else 0.0]
        episode.hist_data["rate_crash"] = [1.0 if reason == "ground_crash" else 0.0]
        episode.hist_data["rate_oob"] = [1.0 if reason == "out_of_bounds" else 0.0]
        episode.hist_data["rate_timeout"] = [1.0 if reason == "timeout" else 0.0]

if __name__ == "__main__":
    # 1. 初始化 Ray 引擎
    ray.init(
        _system_config={
            "object_timeout_milliseconds": 10000, # 延长对象超时容忍
        }
    )

    # 2. 注册环境名称
    env_name = "drone_1v1_mappo_env"
    register_env(env_name, env_creator)

    # 向 RLlib 注册自定义的 MAPPO 模型
    ModelCatalog.register_custom_model("mappo_centralized_critic", MAPPOModel)

    # 动态获取空间维度
    temp_env = env_creator({})
    obs_space = temp_env.observation_spaces["attacker_0"]
    act_space = temp_env.action_spaces["attacker_0"]
    print(f"检测到环境观测空间: {obs_space}, 动作空间维度: {act_space.shape}")

    # 3. 核心算法配置 (PPOConfig)
    config = (
        PPOConfig()
        .environment(env=env_name)
        .framework("torch") # 必须指定使用 PyTorch
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
        .env_runners(
            num_env_runners=4,
            sample_timeout_s=300,       # 将超时容忍度从默认的 60 秒延长到 5 分钟
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

            # 在 Phase 1 阶段，只训练攻击机的大脑，目标机大脑完全冻结不参与计算
            policies_to_train=["policy_attacker"]
        )
        
        # 5. 神经网络结构 (Net Arch)
        .training(
            model={"custom_model": "mappo_centralized_critic"},
            train_batch_size=8192,
            minibatch_size=1024,
            lr=5e-5,
            entropy_coeff=0.01,
            clip_param=0.2, # PPO Actor 截断
            vf_clip_param=1000.0, # 大幅放宽 Critic 网络的截断，防止价值网络窒息
            gamma=0.99,         # 折扣因子 (越大越看重长期收益)
            lambda_=0.95,        # GAE 参数 (默认 0.95)
            kl_coeff=0.2,        # KL 散度惩罚系数 (默认 0.2)
        )
    )

    # 6. 构建算法对象
    print("正在构建 RLlib MAPPO 算法对象，请稍候...")
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
    OLD_CHECKPOINT = os.path.abspath("./marl_runs/mappo_run_0603_1515/checkpoints/checkpoint_stage_2_to_3_iter_574" )

    if os.path.exists(OLD_CHECKPOINT):
        print(f"正在恢复旧模型记忆: {OLD_CHECKPOINT}")
        algo.restore(OLD_CHECKPOINT)

        # ==================== 新增：清除旧的优化器状态，防止维度冲突 ====================
        print("正在清除优化器历史动量 (Amnesia Protocol)...")
        def reset_optimizer_state(env_runner):
            # 获取攻击机的策略网络
            policy = env_runner.get_policy("policy_attacker")
            if policy and hasattr(policy, "_optimizers"):
                for opt in policy._optimizers:
                    # opt.state 是一个字典，里面存着 exp_avg 等历史动量。
                    # 直接 clear() 清空它，PyTorch 会在下一步用新的 13 维权重自动重新初始化它！
                    opt.state.clear()
                    
        # 利用 RLlib 的穿透机制，让所有并行的 Worker 都清空自己的优化器缓存
        algo.env_runner_group.foreach_env_runner(reset_optimizer_state)
        # =================================================================================
        
    else:
        print("未发现旧模型，将从随机初始化开始全新训练。")

    tb_writer = SummaryWriter(log_dir=PROJECT_ROOT)

    # ====================================================================
    # 初始化测试环境与全局课程变量
    # ====================================================================
    TEST_ENV = Drone1v1MARLEnv(gui=False)
    CURRENT_STAGE = 2          # 假设你当前是从 Stage 2 继续训练
    EVAL_INTERVAL = 10         # 每训练 10 次迭代，进行一次确定性压测
    TEST_EPISODES = 50         # 每次压测 50 局
    TARGET_SUCCESS_RATE = 0.75 # 晋级阈值：实测胜率达到 75% 升阶

    # 初始化时强制对齐全军的 Stage
    algo.env_runner_group.foreach_env(
        lambda env: env.set_curriculum_stage(CURRENT_STAGE)
    )
    TEST_ENV.set_curriculum_stage(CURRENT_STAGE)
    print(f"已强制全军（包含所有 Worker）进入初始阶段：Stage {CURRENT_STAGE}")
    # ====================================================================

    # 7. 开始训练循环
    TRAIN_ITERATIONS = 500
    best_success_rate = -0.01
    best_checkpoint_path = None    
    global_episodes = 0  # 全局回合计数器 

    # ================= 新增：动态学习率控制变量 =================
    CURRENT_LR = 5e-5      # 初始学习率 (与 config 中的 lr 保持一致)
    MIN_LR = 5e-6          # 学习率下限 (十分之一)，防止模型彻底停止学习
    DECAY_FACTOR = 0.98    # 每次衰减系数 (胜率达标时，当前 LR * 0.98)
    # ============================================================

    print("==================================")
    print("开始 MAPPO 多智能体 1v1 空战对抗训练！")
    print("提示：在终端按下 【Ctrl + C】 可随时安全终止训练并保存模型！")
    print("==================================")

    try: 
        for i in range(TRAIN_ITERATIONS): # 每一次迭代为train_batch_size
            # step() 会让所有 worker 跑环境，收集数据，更新神经网络，然后返回统计信息
            result = algo.train()

            # 【关键修复】提取大脑中真实的迭代年龄，代替外部的 i
            real_iter = result["training_iteration"]

            # 尝试从 env_runners 中获取数据，如果没有则退回使用 result 本身
            stats = result.get("env_runners", result)
            policy_rewards = stats.get("policy_reward_mean", {})
            
            # 打印双方的平均奖励，观察博弈胜负手
            reward_A = policy_rewards.get("policy_attacker", 0.0)
            reward_E = policy_rewards.get("policy_evader", 0.0)

            # 提取总训练步数和本轮完成的回合数
            total_steps = result.get("num_env_steps_trained", 0)

            # 获取本轮准确的回合数
            episodes_this_iter = stats.get("episodes_this_iter", 0)

            # 精准提取本轮迭代的真实统计
            hist_stats = stats.get("hist_stats", {})

            # 提取历史记录列表 (RLlib 默认保留最近的 100 局)
            success_list = hist_stats.get("rate_success", [])
            crash_list   = hist_stats.get("rate_crash", [])
            oob_list     = hist_stats.get("rate_oob", [])
            timeout_list = hist_stats.get("rate_timeout", [])

            # 利用切片 (Slicing) 强制只取最后 N 局的数据
            def calc_iter_mean(lst, num_recent):
                if num_recent <= 0 or not lst:
                    return 0.0
                # 取列表最后的 num_recent 个元素
                recent_lst = lst[-num_recent:]
                return sum(recent_lst) / len(recent_lst)

            success_rate = calc_iter_mean(success_list, episodes_this_iter)
            crash_rate   = calc_iter_mean(crash_list, episodes_this_iter)
            oob_rate     = calc_iter_mean(oob_list, episodes_this_iter)
            timeout_rate = calc_iter_mean(timeout_list, episodes_this_iter)

            # 提取策略熵 (Entropy) 
            learner_info = result.get("info", {}).get("learner", {})
            attacker_learner = learner_info.get("policy_attacker", {})
            
            # 兼容 RLlib 的不同嵌套层级
            learner_stats = attacker_learner.get("learner_stats", attacker_learner)
            
            # 提取 Entropy
            entropy = learner_stats.get("entropy", 0.0)

            # ================= 新增：基于胜率的动态学习率衰减 =================
            # 当真实胜率突破 50%，且还没跌破下限时，开始温和衰减学习率
            if success_rate > 0.50 and CURRENT_LR > MIN_LR:
                CURRENT_LR = max(MIN_LR, CURRENT_LR * DECAY_FACTOR)

                # 【核心关键】：必须将新的学习率穿透同步给底层的 PyTorch 优化器
                def set_lr(env_runner):
                    policy = env_runner.get_policy("policy_attacker")
                    if policy and hasattr(policy, "_optimizers"):
                        for opt in policy._optimizers:
                            for param_group in opt.param_groups:
                                param_group["lr"] = CURRENT_LR

                # 广播给所有的 Worker 进程
                algo.env_runner_group.foreach_env_runner(set_lr)
            # =================================================================
            
            print(f"迭代 {real_iter:03d} | "
                  f"奖励(主/敌): {reward_A:6.1f} / {reward_E:6.1f} | "
                  f"本轮真实终局 -> 击杀:{success_rate*100:5.1f}% | 坠地:{crash_rate*100:5.1f}% | 越界:{oob_rate*100:5.1f}% | 超时:{timeout_rate*100:5.1f}% | "
                  f"本轮局数: {episodes_this_iter:3d} | "
                  f"熵: {entropy:.4f} | "
                  f"总训练步数: {total_steps}")
            
            # 写入宏观平均曲线 (横坐标为 Iteration)
            tb_writer.add_scalar("1_Rewards/Attacker", reward_A, real_iter)
            tb_writer.add_scalar("1_Rewards/Evader", reward_E, real_iter)
            
            tb_writer.add_scalar("2_Combat_Rates/Success_Kill", success_rate * 100, real_iter)
            tb_writer.add_scalar("2_Combat_Rates/Ground_Crash", crash_rate * 100, real_iter)
            tb_writer.add_scalar("2_Combat_Rates/Out_of_Bounds", oob_rate * 100, real_iter)
            tb_writer.add_scalar("2_Combat_Rates/Timeout", timeout_rate * 100, real_iter)
            tb_writer.add_scalar("5_Network_Stats/Entropy", entropy, real_iter)
            tb_writer.add_scalar("5_Network_Stats/Learning_Rate", CURRENT_LR, real_iter)
            
            # ====================================================================
            # 植入实测与晋级循环
            # ====================================================================
            is_just_upgraded = False  # 【新增】初始化拦截标识

            if (i + 1) % EVAL_INTERVAL == 0:
                print(f"\n{'='*45}")
                print(f"正在进行 Stage {CURRENT_STAGE} 确定性高压测试 ({TEST_EPISODES} 局)...")
                
                success_count = 0
                for _ in range(TEST_EPISODES):
                    obs, info = TEST_ENV.reset()
                    terminated = {"__all__": False}
                    truncated = {"__all__": False}
                    final_reason = "timeout"
                    
                    while not (terminated["__all__"] or truncated["__all__"]):
                        # 开启 explore=False 关闭高斯噪声，获取确定性最优动作
                        action_A = algo.compute_single_action(obs["attacker_0"], policy_id="policy_attacker", explore=False)
                        
                        # 构建 actions 字典
                        actions = {"attacker_0": action_A}
                        if "evader_0" in obs:
                            action_E = algo.compute_single_action(obs["evader_0"], policy_id="policy_evader", explore=False)
                            actions["evader_0"] = action_E
                            
                        obs, rewards, terminated, truncated, infos = TEST_ENV.step(actions)
                        
                        if "attacker_0" in infos and "reason" in infos["attacker_0"]:
                            final_reason = infos["attacker_0"]["reason"]
                    
                    if final_reason == "success":
                        success_count += 1
                
                eval_success_rate = success_count / TEST_EPISODES
                print(f"--> 实测完成！真实击杀率: {eval_success_rate*100:.1f}% ({success_count}/{TEST_EPISODES})")
                
                # 将真实的实测胜率写入 TensorBoard
                tb_writer.add_scalar("2_Combat_Rates/Eval_Success_Rate", eval_success_rate * 100, real_iter)
                
                # 判定是否满足晋级条件！
                if eval_success_rate >= TARGET_SUCCESS_RATE and CURRENT_STAGE < 3:
                    old_stage = CURRENT_STAGE
                    CURRENT_STAGE += 1
                    print(f"突破瓶颈！真实胜率达标，全军晋级到 Stage {CURRENT_STAGE}！")
                    
                    # ================= 新增核心逻辑：里程碑保存与打分重置 =================
                    # 1. 立即保存这个具有纪念意义的转阶段模型 (命名加上特殊的 Stage 标记)
                    transition_save_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_stage_{old_stage}_to_{CURRENT_STAGE}_iter_{real_iter:03d}")
                    algo.save(transition_save_path)
                    print(f"--> [里程碑] 完美通过 Stage {old_stage}，毕业模型已保存至: {transition_save_path}")

                    # 2. 强制重置最佳胜率历史记录
                    # 防止因为上一阶段的“高分滤镜”，导致下一阶段艰难爬坡时无法触发最优模型保存机制
                    best_success_rate = -0.01 
                    is_just_upgraded = True   # 【新增】标记本轮发生了阶级跨越，阻止旧数据污染新基线
                    print(f"--> [系统重置] 已清空上一阶段最高胜率记录，准备记录 Stage {CURRENT_STAGE} 的新征程！")
                    # ====================================================================
                    
                    # 将最新难度广播给底层所有并行搜集数据的 Workers
                    algo.env_runner_group.foreach_env(
                        lambda env: env.set_curriculum_stage(CURRENT_STAGE)
                    )
                    # 同时更新测试环境的难度
                    TEST_ENV.set_curriculum_stage(CURRENT_STAGE)
                    
                print(f"{'='*45}\n")
            # ====================================================================
            
            # RLlib 默认会将 policy 奖励存为 "policy_{policy_id}_reward"
            a_rewards_hist = hist_stats.get("policy_policy_attacker_reward", [])
            e_rewards_hist = hist_stats.get("policy_policy_evader_reward", [])
            
            # 你在 callback 里记录的 custom_metrics 也会原封不动保存在这里
            success_hist = hist_stats.get("rate_success", [])

            # 将当前难度阶段画到图表里
            # 【修复】将 current_stage = ... 删除，直接用大写的 CURRENT_STAGE
            tb_writer.add_scalar("5_Network_Stats/Curriculum_Stage", CURRENT_STAGE, real_iter)

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
            
            tb_writer.flush() # 强制立刻写盘，绝不缓存延迟！
            
            # 保存最高成功率模型
            # 【修改】加入 and not is_just_upgraded，拦截晋级当轮的幽灵数据
            if success_rate > best_success_rate and not is_just_upgraded:
                # 针对 0% 的初次保存做个特殊打印，后面的正常打印提升比例
                if best_success_rate < 0:
                    print(f"建立初始战术基线！当前成功率：{success_rate * 100:.1f}%")
                else:
                    print(f"战术突破！发现新的最高成功率：{best_success_rate * 100:.1f}% -> {success_rate * 100:.1f}%")
                
                best_success_rate = success_rate
                
                # 构建带有迭代次数的新文件夹名称
                new_best_dir = os.path.join(CHECKPOINT_DIR, f"checkpoint_best_iter_{real_iter:03d}")
                
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
                current_save_path = os.path.join(CHECKPOINT_DIR,f"checkpoint_{real_iter:06d}")
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
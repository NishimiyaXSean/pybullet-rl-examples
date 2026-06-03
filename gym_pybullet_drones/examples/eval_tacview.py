import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 引入 RLlib 的策略加载基类
from ray.rllib.policy.policy import Policy
from ray.rllib.models import ModelCatalog
from mappo_model import MAPPOModel
from marl_env import Drone1v1MARLEnv

def evaluate_and_record():
    print("启动 Ray RLlib -> Tacview 评估录像程序...")
    
    # 1. 实例化环境，强制开启 Tacview 记录
    # 评估时关闭 GUI (gui=False)，后台纯跑数据，生成极快
    env = Drone1v1MARLEnv(gui=False, record_tacview=True)

    env.set_curriculum_stage(1) 
    print(f"当前录像环境已设置为：Stage {env.curriculum_stage}")

    ModelCatalog.register_custom_model("mappo_centralized_critic", MAPPOModel)
    
    # 2. 尝试加载 RLlib 策略文件夹
    # 确保 policy_attacker 和 policy_evader 文件夹就在当前运行目录下
    attacker_dir = os.path.abspath("./marl_runs/mappo_run_0602_1521/checkpoints/checkpoint_best_iter_123/policies/policy_attacker")
    evader_dir = os.path.abspath("./marl_runs/mappo_run_0602_1521/checkpoints/checkpoint_best_iter_123/policies/policy_evader")

    policies = {}
    use_model = False
    
    try:
        policies["attacker_0"] = Policy.from_checkpoint(attacker_dir)
        policies["evader_0"] = Policy.from_checkpoint(evader_dir)
        print("成功加载 RLlib 策略权重！")
        use_model = True
    except Exception as e:
        print(f"策略加载失败，原因: {e}")
        print("将使用【随机动作】进行回退测试...")
        use_model = False

    # 3. 初始化环境
    obs_dict, info_dict = env.reset()
    done = False
    step_count = 0

    print("正在生成空战轨迹...")
    
    # 4. 运行单局推演
    while not done:
        actions = {}
        
        # 遍历当前存活的智能体生成动作
        for agent in env.agents:
            if use_model and agent in policies:
                # RLlib 的前向推理接口是 compute_single_action
                # 它返回一个元组：(action, state_outs, info)
                # 我们只需要第一个元素 action
                action, _, _ = policies[agent].compute_single_action(
                    obs=obs_dict[agent], 
                    explore=False  # 评估模式：关闭探索，使用完全确定性的动作
                )
                actions[agent] = action
            else:
                # 随机动作保底测试
                actions[agent] = env.action_spaces[agent].sample()

        # 步进环境
        obs_dict, rewards, terminations, truncations, infos = env.step(actions)
        
        # 判断全局是否结束 (只要有一方坠毁、击杀或超时，__all__ 即为 True)
        done = terminations.get("__all__", False) or truncations.get("__all__", False)
        step_count += 1

    print(f"回合结束，共执行了 {step_count} 个宏观决策步。")
    print(f"Tacview 日志文件已生成。")

if __name__ == "__main__":
    evaluate_and_record()
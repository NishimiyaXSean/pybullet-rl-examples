import numpy as np
import gymnasium as gym
from pettingzoo import ParallelEnv
import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class Drone1v1MARLEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "drone_1v1_v0"}

    def __init__(self, gui=False):
        super().__init__()
        # 1. 明确定义智能体身份 (PettingZoo 规范核心)
        self.possible_agents = ["attacker_0", "evader_0"]
        # 在运行中，如果某架飞机坠毁，它会从 self.agents 列表中被移除
        self.agents = self.possible_agents[:]

        # 2. 实例化底层物理引擎 (CtrlAviary)
        # 将两架飞机分别放置在场地的对角线位置，拉开初始距离
        init_xyzs = np.array([
            [-5.0, -5.0, 2.0],  # attacker_0 的初始位置 (ID: 0)
            [ 5.0,  5.0, 3.0]   # evader_0   的初始位置 (ID: 1)
        ])
        
        self.pyb_env = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=2,           
            initial_xyzs=init_xyzs,
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=60,
            gui=gui,
        )
        
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # 隐藏 PyBullet 默认的左右侧边栏和参数面板
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1) # 开启高质量阴影

        # 两台无人机各自独立的 PID 控制器
        self.pids = {
            "attacker_0": DSLPIDControl(drone_model=DroneModel.CF2X),
            "evader_0": DSLPIDControl(drone_model=DroneModel.CF2X)
        }
        

        self.CTRL_FREQ = 60
        self.is_manual_mode = False

        # 3. 字典化的观测空间与动作空间
        # 动作空间：两者均为 11 维离散动作 (BFM)
        self.action_spaces = {
            agent: gym.spaces.Discrete(11) 
            for agent in self.possible_agents
        }
        
        # 观测空间：各自的第一人称视角 (原为19维，可根据后续设计调整)
        self.observation_spaces = {
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(19,), dtype=np.float32)
            for agent in self.possible_agents
        }

    # PettingZoo 强制要求提供 action_space 和 observation_space 的读取接口
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        """
        环境重置，必须返回两个字典：obs_dict, info_dict
        """
        # 重置存活列表
        self.agents = self.possible_agents[:]
        
        # 重置底层物理引擎
        raw_obs, _ = self.pyb_env.reset()
        
        # 这里需要获取两架飞机的底层状态
        # attacker_state = self.pyb_env._getDroneStateVector(0)
        # evader_state = self.pyb_env._getDroneStateVector(1)
        
        # TODO: 编写计算各自身份视角的 _compute_obs 函数
        # 暂时用零矩阵代替跑通流程
        obs_dict = {
            "attacker_0": np.zeros(19, dtype=np.float32),
            "evader_0": np.zeros(19, dtype=np.float32)
        }
        
        info_dict = {agent: {} for agent in self.agents}
        
        return obs_dict, info_dict

    def step(self, actions):
        """
        接受包含多个智能体动作的字典，执行物理步进，返回五个字典
        """
        # 如果所有飞机都坠毁了，提前返回空字典 (PettingZoo 保护机制)
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # 1. 动作解析与物理驱动脚手架
        # 原来的 step 里包含 frame_skip 的 repeat 循环和 PID 跟随逻辑
        # 在这里，需要：
        #   - 读取 actions["attacker_0"] 和 actions["evader_0"]
        #   - 将离散动作转化为两者的期望偏航角(yaw)和期望速度
        #   - 丢进 self.pids 分别计算 RPM
        #   - 调用 self.pyb_env.step(np.vstack([rpm_attacker, rpm_evader]))
        
        # ==========================================
        # TODO: 这里将填入双人 PID 与 BFM 解析逻辑
        # ==========================================

        # 2. 状态更新、碰撞检测与奖励计算
        # 暂时用 Dummy 数据填充
        obs_dict = {
            "attacker_0": np.zeros(19, dtype=np.float32),
            "evader_0": np.zeros(19, dtype=np.float32)
        }
        
        rewards_dict = {
            "attacker_0": 0.0,
            "evader_0": 0.0
        }
        
        terminations_dict = {
            "attacker_0": False,
            "evader_0": False
        }
        
        truncations_dict = {
            "attacker_0": False,
            "evader_0": False
        }
        
        infos_dict = {
            "attacker_0": {},
            "evader_0": {}
        }
        
        # 3. 清理死亡的智能体 
        # 如果某一方终止了，必须把它从 self.agents 列表中剔除
        self.agents = [
            agent for agent in self.agents
            if not (terminations_dict[agent] or truncations_dict[agent])
        ]

        return obs_dict, rewards_dict, terminations_dict, truncations_dict, infos_dict
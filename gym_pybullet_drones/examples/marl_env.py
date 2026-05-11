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
        核心物理步进函数，接收字典 actions = {"attacker_0": a1, "evader_0": a2}
        """
        # 如果所有飞机都坠毁了，提前返回空字典 (PettingZoo 保护机制)
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # 统一决策频率 (Frame Skip)
        # 强制规定 AI 每 0.2 秒做一次决策 (在 60Hz 的底层频率下，相当于推进 12 帧)
        AI_DECISION_DT = 0.2 
        dynamic_frame_skip = int(AI_DECISION_DT * self.CTRL_FREQ)
        dt = 1 / self.CTRL_FREQ

        total_rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        infos = {agent: {} for agent in self.agents}

        # 提取两架飞机的初始状态 (用于后续计算奖励和碰撞)
        attacker_id = 0
        evader_id = 1

        for _ in range(dynamic_frame_skip):
            # 获取最新物理状态
            attacker_state = self.pyb_env._getDroneStateVector(attacker_id)
            evader_state = self.pyb_env._getDroneStateVector(evader_id)
            
            attacker_pos = attacker_state[0:3]
            evader_pos = evader_state[0:3]
            dist = np.linalg.norm(attacker_pos - evader_pos)

            # --- 动作解码与 PID 预处理 ---
            # 建立 RPM 数组，准备传给 PyBullet (维度: 2台飞机 x 4个电机)
            rpms = np.zeros((2, 4))
            
            for i, agent in enumerate(["attacker_0", "evader_0"]):
                if agent not in actions: # 如果这架飞机已经坠毁，跳过
                    continue
                    
                action_int = int(actions[agent])
                n_x, n_n, mu = self.bfm_action_mapping[action_int]
                
                # 这里引入非对称设计：目标机(evader)的速度稍微慢一点
                speed_multiplier = 0.8 if agent == "evader_0" else 1.0
                vx = (1.5 + n_x * 0.8) * speed_multiplier
                vy = 0.0
                vz = (n_n * np.cos(mu) - 1.0) * 0.15
                
                yaw_rate = n_n * np.sin(mu) * 0.5
                target_vel_local = np.array([vx, vy, vz])
                
                # 累加 Yaw 角度
                self.target_yaws[agent] += yaw_rate * dt

                # 生成一个没有俯仰(Pitch)和滚转(Roll)的纯净偏航姿态
                pilot_quat = p.getQuaternionFromEuler([0, 0, self.target_yaws[agent]])
                
                vel_world, _ = p.multiplyTransforms([0,0,0], pilot_quat, target_vel_local, [0,0,0,1])
                vel_world = np.array(vel_world)

                self.user_input_pos[agent] += vel_world * dt
                
                # 高度限制防钻地
                self.user_input_pos[agent][2] = np.clip(self.user_input_pos[agent][2], 1.0, 10.0)
                
                # PID 平滑追踪
                self.current_target_pos[agent] = self.current_target_pos[agent] * (1 - self.SMOOTH_FACTOR) + self.user_input_pos[agent] * self.SMOOTH_FACTOR
                
                # 计算这台飞机的 RPM
                agent_state = attacker_state if i == 0 else evader_state
                rpm, _, _ = self.pids[agent].computeControl(
                    control_timestep=dt,
                    cur_pos=agent_state[0:3],       
                    cur_quat=agent_state[3:7],     
                    cur_vel=agent_state[10:13],     
                    cur_ang_vel=agent_state[13:16],
                    target_pos=self.current_target_pos[agent], 
                    target_vel=vel_world,
                    target_rpy=np.array([0, 0, self.target_yaws[agent]])
                )
                rpms[i, :] = rpm

            # --- 底层物理步进 ---
            # 把两台飞机的 RPM 打包丢给物理引擎
            self.pyb_env.step(rpms)
            self.step_counter += 1

            # --- 碰撞与终止条件检测 (在每一小帧都要检测) ---
            new_attacker_state = self.pyb_env._getDroneStateVector(attacker_id)
            new_evader_state = self.pyb_env._getDroneStateVector(evader_id)
            new_dist = np.linalg.norm(new_attacker_state[0:3] - new_evader_state[0:3])

            # 1. 动能撞击 / 击杀成功
            if new_dist < 0.15:
                if "attacker_0" in actions: total_rewards["attacker_0"] += 300.0
                if "evader_0" in actions: total_rewards["evader_0"] -= 300.0
                terminations["attacker_0"] = True
                terminations["evader_0"] = True
                break # 直接结束本轮 AI 决策的 repeat 循环

            # 2. 地板/天空边界惩罚
            for agent, state in zip(["attacker_0", "evader_0"], [new_attacker_state, new_evader_state]):
                if state[2] < 0.1 or state[2] > 12.0:
                    total_rewards[agent] -= 100.0
                    terminations[agent] = True

            # (此处可插入细粒度奖励：ATA角惩罚、距离拉近奖励等)
            # attacker_reward += (prev_dist - new_dist) * 20
            # evader_reward   += (new_dist - prev_dist) * 10

        # --- 退出 Frame Skip 循环，结算当前决策步的最终结果 ---
        
        # 判断是否超时 (Truncation)
        if (self.step_counter / self.CTRL_FREQ) > self.EPISODE_LEN_SEC:
            truncations["attacker_0"] = True
            truncations["evader_0"] = True
        
        # 计算最新的观测值
        observations = {}
        for agent in self.agents:
            if not terminations[agent]:
                observations[agent] = self._compute_obs(agent)
            else:
                # 如果飞机死了，按 PettingZoo 规矩传零向量
                observations[agent] = np.zeros(19, dtype=np.float32)

        # 必须清理掉本回合死亡的智能体
        self.agents = [
            a for a in self.agents
            if not (terminations[a] or truncations[a])
        ]

        return observations, total_rewards, terminations, truncations, infos
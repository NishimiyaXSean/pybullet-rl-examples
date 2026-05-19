import numpy as np
import gymnasium as gym
import pybullet as p
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class Drone1v1MARLEnv(MultiAgentEnv):
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
            [-2000.0, -2000.0, 3000.0],  # attacker_0 的初始位置 (ID: 0)
            [ 2000.0,  2000.0, 3000.0]   # evader_0   的初始位置 (ID: 1)
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

        self.CTRL_FREQ = 60
        self.is_manual_mode = False
        self.EPISODE_LEN_SEC = 100 # 回合最大时长
        self.cpa_radius = 150.0     # 近炸引信触发半径

        # --- 战斗机飞行包线参数 (F-16/歼-10 级别模拟) ---
        self.MAX_G = 9.0          # 最大结构过载 (正G)
        self.MIN_G = -3.0         # 最大负过载 (通常远小于正G)
        self.CORNER_SPEED = 150.0 # 角速度 (约 540 km/h)：能拉出最大过载的最低速度
        self.MAX_SPEED = 400.0    # 绝对最大平飞速度 (约 1.2 马赫)
        self.STALL_SPEED = 60.0   # 基础失速速度
        self.g = 9.81             # 重力加速度

        # --- 目标机(Evader)性能缩放系数 ---
        self.EVADER_SPEED_COEFF = 0.625  # 速度系数 (400 * 0.625 = 250 m/s)
        self.EVADER_G_COEFF = 0.555      # 过载系数 (9.0 * 0.555 ≈ 5.0 G)

        # BFM 动作库: {动作编号 : (切向过载 n_x, 法向过载 n_n, 滚转角 mu)}
        self.bfm_action_mapping = {
            0:  ( 0,  1,  0.0),            # a1: 匀速直飞
            1:  ( 2,  1,  0.0),            # a2: 加速直飞
            2:  (-2,  1,  0.0),            # a3: 减速直飞
            3:  ( 0,  8,  0.0),            # a4: 跃升
            4:  ( 0, -8,  0.0),            # a5: 俯冲
            5:  ( 0,  8,  np.pi / 3.0),    # a6: 左转跃升
            6:  ( 0, -8, -np.pi / 3.0),    # a7: 右转俯冲
            7:  ( 0,  8, -np.pi / 3.0),    # a8: 右转跃升
            8:  ( 0, -8,  np.pi / 3.0),    # a9: 左转俯冲
            9:  ( 0,  2, -np.pi / 3.0),    # a10: 右转
            10: ( 0,  2,  np.pi / 3.0)     # a11: 左转
        }

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

        # 视觉与运镜初始化
        self.camera_mode = 1  # 默认相机视角 (1:智能追尾)
        self.hud_text_id = -1
        self.fuze_obj_id = -1
        self.last_draw_pos = np.zeros(3)
        self.last_target_draw_pos = np.zeros(3)
        self.cam_pos = np.zeros(3)

    def reset(self, seed=None, options=None):
        """
        环境重置，必须返回两个字典：obs_dict, info_dict
        """
        # 重置存活列表
        self.agents = self.possible_agents[:]

        # 动态生成对角线象限的随机出生点
        # 1. 随机决定主机的象限符号 (1 或 -1)
        sign_x = np.random.choice([-1, 1])
        sign_y = np.random.choice([-1, 1])

        # 2. 在该象限内，生成 2000 到 4000 米的初始距离
        attacker_x = sign_x * np.random.uniform(2000.0, 4000.0)
        attacker_y = sign_y * np.random.uniform(2000.0, 4000.0)
        attacker_z = np.random.uniform(3000.0, 5000.0) 

        # 3. 目标机强制取相反符号，确保永远出生在对角象限！
        evader_x = -sign_x * np.random.uniform(2000.0, 4000.0)
        evader_y = -sign_y * np.random.uniform(2000.0, 4000.0)
        evader_z = np.random.uniform(2000.0, 4000.0)
        self.evader_initial_z = evader_z

        # 组合成新的初始坐标数组
        new_init_xyzs = np.array([
            [attacker_x, attacker_y, attacker_z],
            [evader_x, evader_y, evader_z]
        ])

        # 覆盖 gym-pybullet-drones 底层环境缓存的初始坐标
        self.pyb_env.INIT_XYZS = new_init_xyzs
        
        # 重置底层物理引擎
        raw_obs, _ = self.pyb_env.reset()

        initial_speed = 150.0  # 设定初始空速为 150 m/s (约 540 km/h)

        # 替换 reset 函数中原本的初始姿态和速度赋值：
        for i, agent in enumerate(self.agents):
            initial_pos = new_init_xyzs[i]
            pyb_id = self.pyb_env.DRONE_IDS[i] if hasattr(self.pyb_env, 'DRONE_IDS') else self.pyb_env.drone_ids[i]
            
            # 修复：计算指向原点 (0,0) 的偏航角
            dx = -initial_pos[0]
            dy = -initial_pos[1]
            yaw = np.arctan2(dy, dx)
            
            # 根据真实偏航角分解 X 和 Y 方向的初始速度
            init_vel = [initial_speed * np.cos(yaw), initial_speed * np.sin(yaw), 0.0]
            init_quat = p.getQuaternionFromEuler([0, 0, yaw])
            
            p.resetBasePositionAndOrientation(pyb_id, initial_pos, init_quat, physicsClientId=self.pyb_env.CLIENT)
            p.resetBaseVelocity(pyb_id, linearVelocity=init_vel, physicsClientId=self.pyb_env.CLIENT)

        if hasattr(self.pyb_env, '_updateAndStoreKinematicInformation'):
            self.pyb_env._updateAndStoreKinematicInformation()
        
        # 初始化时间步与两架飞机的局部追踪变量
        self.step_counter = 0  # 留着给底层备用
        self.macro_step = 0    # 真正的宏观决策步数
        self.last_actions = {agent: 0 for agent in self.agents}
    
        # 计算开局时的初始距离 (用于第一帧的奖励计算基准)
        attacker_pos = self.pyb_env._getDroneStateVector(0)[0:3]
        evader_pos = self.pyb_env._getDroneStateVector(1)[0:3]
        self.prev_dist = np.linalg.norm(attacker_pos - evader_pos)
    
        obs_dict = {
            "attacker_0": self._compute_obs("attacker_0"),
            "evader_0": self._compute_obs("evader_0")
        }
        
        info_dict = {agent: {} for agent in self.agents}
        
        # 3D 视觉场景构建 (仅在开启 GUI 时生效)
        self.last_draw_pos = attacker_pos.copy()
        self.last_target_draw_pos = evader_pos.copy()
        self.cam_pos = attacker_pos.copy()

        if self.pyb_env.GUI:
            p.removeAllUserDebugItems(physicsClientId=self.pyb_env.CLIENT) # 清理上一局的残留线条
            
            # --- 新增：高空战术参考网格 ---
            # 设定在 3000米、6000米、9000米 绘制三层不同颜色的半透明网格
            grid_altitudes = {
                3000.0: [0.0, 0.5, 1.0],  # 浅蓝色
                6000.0: [0.0, 1.0, 0.5],  # 青绿色
                9000.0: [1.0, 0.5, 0.0]   # 橙色
            }
            
            grid_size = 10000.0 # 网格覆盖范围 (正负 10km)
            grid_step = 2000.0  # 每 2km 画一条线
            
            for z, color in grid_altitudes.items():
                # 沿着 X 轴画线
                for y in np.arange(-grid_size, grid_size + 1, grid_step):
                    p.addUserDebugLine([-grid_size, y, z], [grid_size, y, z], color, 1.0, 0, physicsClientId=self.pyb_env.CLIENT)
                # 沿着 Y 轴画线
                for x in np.arange(-grid_size, grid_size + 1, grid_step):
                    p.addUserDebugLine([x, -grid_size, z], [x, grid_size, z], color, 1.0, 0, physicsClientId=self.pyb_env.CLIENT)
                    
            # 为目标机生成一个半透明的近炸引信杀伤圈
            fuze_v_id = p.createVisualShape(p.GEOM_SPHERE, radius=self.cpa_radius, rgbaColor=[1, 0.5, 0, 0.25])
            self.fuze_obj_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=fuze_v_id, basePosition=evader_pos, physicsClientId=self.pyb_env.CLIENT)

            z_offset = 0.05
            # 绘制主坐标轴
            p.addUserDebugLine([-15, 0, z_offset],[15, 0, z_offset], [1, 0, 0], 4, 0, physicsClientId=self.pyb_env.CLIENT)
            p.addUserDebugLine([0, -15, z_offset], [0, 15, z_offset],[0, 1, 0], 4, 0, physicsClientId=self.pyb_env.CLIENT)
            p.addUserDebugLine([0, 0, z_offset],[0, 0, 15], [0, 0.5, 1], 4, 0, physicsClientId=self.pyb_env.CLIENT)
            
        return obs_dict, info_dict
    
    def _compute_obs(self, agent):
        """
        计算指定智能体（agent）的第一人称局部观测值
        """
        # 1. 确定“我”和“敌机”的底层物理 ID
        my_id = 0 if agent == "attacker_0" else 1
        enemy_id = 1 - my_id  # 对方的 ID

        # 2. 获取双方的绝对物理状态
        my_state = self.pyb_env._getDroneStateVector(my_id)
        enemy_state = self.pyb_env._getDroneStateVector(enemy_id)

        my_pos = my_state[0:3]
        my_quat = my_state[3:7]         # 自身四元数
        my_rpy = my_state[7:10]         # 自身姿态 (Roll, Pitch, Yaw)
        my_vel = my_state[10:13]        # 自身速度
        my_ang_vel = my_state[13:16]    # 自身角速度
        my_z_height = my_state[2]       # 自身绝对高度

        enemy_pos = enemy_state[0:3]
        enemy_quat = enemy_state[3:7]   # 敌机四元数
        enemy_vel = enemy_state[10:13]

        # --- 新增：计算 3D 战术几何特征 (ATA, AA, HCA) ---
        # 1. 提取双方的机头朝向向量 (X轴正方向)
        # PyBullet 旋转矩阵解析：[R00, R01, R02, R10, R11, R12, R20, R21, R22]
        # X轴方向的世界坐标系向量即为矩阵的第一列 [R00, R10, R20]
        rot_mat_my = p.getMatrixFromQuaternion(my_quat)
        my_forward = np.array([rot_mat_my[0], rot_mat_my[3], rot_mat_my[6]])

        rot_mat_enemy = p.getMatrixFromQuaternion(enemy_quat)
        enemy_forward = np.array([rot_mat_enemy[0], rot_mat_enemy[3], rot_mat_enemy[6]])

        # 2. 计算视线向量 (Line of Sight, LOS)
        los_vec = enemy_pos - my_pos
        dist = np.linalg.norm(los_vec)
        los_dir = los_vec / dist if dist > 1e-6 else my_forward

        # 3. 计算战术夹角的余弦值 [-1, 1]
        # ATA (天线偏角): 我的机头指向 vs 视线方向 (1表示完美瞄准)
        cos_ata = np.clip(np.dot(my_forward, los_dir), -1.0, 1.0)
        
        # AA (方位角): 敌机尾部/机头 vs 视线方向 
        # (1表示我处于敌机正后方完美的6点钟死角，-1表示处于正前方对头)
        cos_aa = np.clip(np.dot(enemy_forward, los_dir), -1.0, 1.0)
        
        # HCA (航向交叉角): 我的机头指向 vs 敌机机头指向 (1表示同向飞行，-1表示对头飞行)
        cos_hca = np.clip(np.dot(my_forward, enemy_forward), -1.0, 1.0)

        tactical_geometry = np.array([cos_ata, cos_aa, cos_hca], dtype=np.float32)
        # ------------------------------------------------

        # 3. 核心坐标转换：构建“我”的机头坐标系 (基于 Yaw)
        # 提取我在这个物理帧的真实朝向
        my_yaw = my_rpy[2] 
        my_quat = p.getQuaternionFromEuler([0, 0, my_yaw])
        _, inv_quat = p.invertTransform([0, 0, 0], my_quat)

        # 4. 计算相对变量，并投影到“我”的第一人称坐标系中
        # --- A. 敌机相对我的位置 ---
        world_rel_pos = enemy_pos - my_pos
        local_rel_pos, _ = p.multiplyTransforms([0, 0, 0], inv_quat, world_rel_pos, [0, 0, 0, 1])
        local_rel_pos = np.array(local_rel_pos) 

        # --- B. 我的局部速度 ---
        local_vel, _ = p.multiplyTransforms([0, 0, 0], inv_quat, my_vel, [0, 0, 0, 1])
        local_vel = np.array(local_vel)

        # --- C. 我的局部角速度 ---
        local_ang_vel, _ = p.multiplyTransforms([0, 0, 0], inv_quat, my_ang_vel, [0, 0, 0, 1])
        local_ang_vel = np.array(local_ang_vel)

        # --- D. 敌机在我的坐标系下的绝对速度 ---
        local_enemy_vel, _ = p.multiplyTransforms([0, 0, 0], inv_quat, enemy_vel, [0, 0, 0, 1])
        local_enemy_vel = np.array(local_enemy_vel)

        # 5. 物理量级缩放 (Pre-normalization) - 防止神经网络梯度爆炸
        MAX_DIST = 10000.0     
        MAX_HEIGHT = 15000.0
        MAX_VEL = 400.0    
        MAX_ANG_VEL = np.pi 

        norm_local_rel_pos = local_rel_pos / MAX_DIST
        norm_local_vel = local_vel / MAX_VEL
        norm_rpy = my_rpy / np.pi                  
        norm_local_ang_vel = local_ang_vel / MAX_ANG_VEL
        norm_z_height = my_z_height / MAX_HEIGHT     
        norm_local_enemy_vel = local_enemy_vel / MAX_VEL
        
        # 6. 拼接 19 维特征数组，形状严丝合缝
        obs_array = np.concatenate([
            norm_local_rel_pos,    # 3维: 敌机相对位置
            norm_local_vel,        # 3维: 我的空速
            norm_rpy,              # 3维: 我的姿态
            norm_local_ang_vel,    # 3维: 我的角速度
            [norm_z_height],       # 1维: 我的高度
            norm_local_enemy_vel,  # 3维: 敌机速度矢量
            tactical_geometry      # 3维: 空战几何角
        ]).astype(np.float32)

        # 裁剪在 [-1.0, 1.0] 范围内
        obs_array = np.clip(obs_array, -1.0, 1.0)

        return obs_array

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

        # 动作连续性惩罚
        for agent, act in actions.items():
            if act != self.last_actions.get(agent, 0):
                # 每次切换动作，扣除一点体力分，逼迫其保持动作连贯
                total_rewards[agent] -= 0.5 
            self.last_actions[agent] = act

        attacker_state_init = self.pyb_env._getDroneStateVector(attacker_id)
        evader_state_init = self.pyb_env._getDroneStateVector(evader_id)
        dist = np.linalg.norm(attacker_state_init[0:3] - evader_state_init[0:3])
        current_micro_dist = dist

        self.macro_step += 1 # 新增：每次 AI 下达指令，宏观步数推进 1 步

        # ================== 新增：绝对信任的本地物理账本 ==================
        # 彻底抛弃每帧从 PyBullet 读取速度的逻辑，防止无人机的空气阻力污染数据！
        trusted_states = {
            "attacker_0": {"pos": attacker_state_init[0:3].copy(), "vel": attacker_state_init[10:13].copy()},
            "evader_0":   {"pos": evader_state_init[0:3].copy(),   "vel": evader_state_init[10:13].copy()}
        }
        # ==================================================================

        for _ in range(dynamic_frame_skip):           
            self.pyb_env.step(np.zeros((2, 4)))
            self.step_counter += 1
            
            for i, agent in enumerate(["attacker_0", "evader_0"]):
                if agent not in actions or terminations[agent]: # 如果这架飞机已经判定死亡，则直接跳过
                    continue

                pyb_id = self.pyb_env.DRONE_IDS[i] if hasattr(self.pyb_env, 'DRONE_IDS') else self.pyb_env.drone_ids[i]
                action_int = int(actions[agent])
                n_x_cmd, n_n_cmd, mu_cmd = self.bfm_action_mapping[action_int]
                
                current_max_speed = self.MAX_SPEED
                current_max_g = self.MAX_G

                if agent == "evader_0":
                    current_max_speed = self.MAX_SPEED * self.EVADER_SPEED_COEFF
                    current_max_g = self.MAX_G * self.EVADER_G_COEFF

                # 干净账本读取状态 
                pos = trusted_states[agent]["pos"]
                vel = trusted_states[agent]["vel"]

                agent_current_z = pos[2]

                # GPWS 近地警告覆盖
                if agent_current_z < 300.0 and n_n_cmd < 0:
                    n_n_cmd = current_max_g
                    mu_cmd = 0.0 # 改平滚转角
                    total_rewards[agent] -= 2.0 * dt

                V = np.linalg.norm(vel)
                if V < 1e-3: V = 1e-3  # 防止除以 0

                # 包线限制 
                # A. 升力限制 (低速时无法拉出大过载，升力与速度的平方成正比)
                # 抛物线方程：当前可用最大过载 = (当前速度 / 角速度)^2 * 最大结构过载
                available_n_lift = ((V / self.CORNER_SPEED) ** 2) * current_max_g
                
                # B. 结构限制 (取升力限制和物理结构强度的较小值)
                actual_max_n = min(current_max_g, available_n_lift)
                actual_min_n = max(self.MIN_G, -available_n_lift)
                
                # C. 强制裁剪法向过载 (n_n)
                n_n = np.clip(n_n_cmd, actual_min_n, actual_max_n)
                mu = mu_cmd

                # D. 切向过载 (加减速) 的简易动力学限制
                n_x = n_x_cmd
                if V > current_max_speed and n_x_cmd > 0:
                    n_x = 0.0  # 超过极速无法继续加速 (阻力壁垒)
                elif V < self.STALL_SPEED and n_x_cmd < 0:
                    n_x = 0.0  # 接近失速时无法继续减速

                # 新增 MARL 学习优化：惩罚“违背物理定律”的意图
                # 如果 AI 给出的指令超出了物理极限被强制裁剪了，必须扣除奖励，逼迫神经网络学习并记忆这架飞机的包线，而不是瞎推杆。
                if abs(n_n_cmd - n_n) > 0.1:
                    total_rewards[agent] -= 1.0 * dt  # "蛮力推杆"惩罚
                
                gamma = np.arcsin(np.clip(vel[2] / V, -1.0, 1.0)) # 航迹俯仰角
                chi = np.arctan2(vel[1], vel[0])                  # 航迹方位角
                V_dot = self.g * (n_x - np.sin(gamma))
                
                # 防止大俯仰角时出现奇点 (gamma 接近 90 度时 cos(gamma) 接近 0)
                cos_gamma = np.cos(gamma) if abs(np.cos(gamma)) > 1e-3 else 1e-3
                
                gamma_dot = (self.g / V) * (n_n * np.cos(mu) - np.cos(gamma))
                chi_dot = (self.g * n_n * np.sin(mu)) / (V * cos_gamma)
                
                # 欧拉积分更新状态
                new_V = V + V_dot * dt
                new_gamma = gamma + gamma_dot * dt
                new_chi = chi + chi_dot * dt
                
                # 将极坐标下的速度转换回 3D 笛卡尔坐标系
                new_vel = np.array([
                    new_V * np.cos(new_gamma) * np.cos(new_chi),
                    new_V * np.cos(new_gamma) * np.sin(new_chi),
                    new_V * np.sin(new_gamma)
                ])
                
                new_pos = pos + new_vel * dt
                
                # 计算新姿态四元数 (根据速度方向和滚转角对齐机头)
                new_quat = p.getQuaternionFromEuler([mu, new_gamma, new_chi])

                # 高度限制与天花板惩罚 
                if agent == "attacker_0":
                    # 我方无人机：触碰天花板时给予持续惩罚，防止利用边界“滑行”
                    if new_pos[2] > 15000.0 :
                        new_pos[2] = 15000.0
                        total_rewards[agent] -= 0.5 * dt  # 累加高度软惩罚
                    # 正常防钻地（不给惩罚，直接限制）
                    elif new_pos[2] < 1.0:
                        new_pos[2] = 1.0
                else:
                    # 目标机：仅保留原有的物理边界限制，不施加任何额外惩罚
                    new_pos[2] = np.clip(new_pos[2], 1.0, 15000.0)

                # ================= 核心修复：更新本地账本并强制洗白 PyBullet =================
                trusted_states[agent]["pos"] = new_pos
                trusted_states[agent]["vel"] = new_vel

                # 强行把洗干净的数据覆盖回被空气阻力弄脏的 PyBullet
                p.resetBasePositionAndOrientation(pyb_id, new_pos, new_quat, physicsClientId=self.pyb_env.CLIENT)
                p.resetBaseVelocity(pyb_id, linearVelocity=new_vel, physicsClientId=self.pyb_env.CLIENT)

            # 更新 gym-pybullet-drones 的内置缓存，保证底层 Observation 读取正确
            if hasattr(self.pyb_env, '_updateAndStoreKinematicInformation'):
                self.pyb_env._updateAndStoreKinematicInformation()

            # 重新提取一次绝对干净的物理状态，用于下方的距离、几何计算和画图
            new_attacker_state = self.pyb_env._getDroneStateVector(attacker_id)
            new_evader_state = self.pyb_env._getDroneStateVector(evader_id)
            
            new_attacker_pos = new_attacker_state[0:3]
            new_evader_pos = new_evader_state[0:3]
            
            # 计算最新的微小帧距离和变化率
            new_dist = np.linalg.norm(new_attacker_pos - new_evader_pos)
            if new_dist < 1e-3:  # 如果一开局距离就=变成 0，说明底层坐标提取发生奇异，强制修正
                new_dist = 1e-3

            # ================= 物理极速限制 =================
            raw_micro_delta = new_dist - current_micro_dist
            micro_delta_dist = np.clip(raw_micro_delta, -20.0, 20.0) 
            current_micro_dist = new_dist

            # 1:1 真实物理平滑渲染与电影级运镜
            if self.pyb_env.GUI:
                import time
                time.sleep(1 / self.CTRL_FREQ)  # 强制同步现实时间
                
                # 实时更新新坐标
                cur_attacker_pos = self.pyb_env._getDroneStateVector(attacker_id)[0:3]
                cur_evader_pos = self.pyb_env._getDroneStateVector(evader_id)[0:3]
                
                # 更新目标机身上的“幽灵引信球”位置
                if self.fuze_obj_id != -1:
                    p.resetBasePositionAndOrientation(self.fuze_obj_id, cur_evader_pos, [0, 0, 0, 1], physicsClientId=self.pyb_env.CLIENT)

                # 键盘运镜切换监听
                keys = p.getKeyboardEvents(physicsClientId=self.pyb_env.CLIENT)
                if ord('1') in keys and keys[ord('1')] & p.KEY_WAS_TRIGGERED: self.camera_mode = 1
                if ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED: self.camera_mode = 2
                if ord('3') in keys and keys[ord('3')] & p.KEY_WAS_TRIGGERED: self.camera_mode = 3
                if ord('4') in keys and keys[ord('4')] & p.KEY_WAS_TRIGGERED: self.camera_mode = 4
                if ord('5') in keys and keys[ord('5')] & p.KEY_WAS_TRIGGERED: self.camera_mode = 5

                # 平滑画出红色(主机)与黄色(目标机)的 3D 轨迹尾迹
                p.addUserDebugLine(self.last_draw_pos, cur_attacker_pos, [1, 0, 0], 2.5, 3.0, physicsClientId=self.pyb_env.CLIENT)
                p.addUserDebugLine(self.last_target_draw_pos, cur_evader_pos, [1, 1, 0], 2.5, 3.0, physicsClientId=self.pyb_env.CLIENT)
                self.last_draw_pos = cur_attacker_pos.copy()
                self.last_target_draw_pos = cur_evader_pos.copy()

                # HUD 文字与几何计算
                dist_cam = np.linalg.norm(cur_attacker_pos - cur_evader_pos)
                dx = cur_attacker_pos[0] - cur_evader_pos[0]
                dy = cur_attacker_pos[1] - cur_evader_pos[1]
                drone_angle = np.degrees(np.arctan2(dy, dx))
                
                vel_norm = np.linalg.norm(self.pyb_env._getDroneStateVector(attacker_id)[10:13])
                hud_text = f"Dist:{dist_cam:.1f}m | ATA:{drone_angle:.0f}deg | Vel:{vel_norm:.1f}m/s"
                
                # 获取主机 ID，用于绑定 HUD 文本跟随
                drone_pyb_id = self.pyb_env.DRONE_IDS[0] if hasattr(self.pyb_env, 'DRONE_IDS') else self.pyb_env.drone_ids[0]
                if self.hud_text_id == -1:
                    self.hud_text_id = p.addUserDebugText(hud_text, [0, 0, 0.8], textColorRGB=[0, 0, 0], textSize=1.5, parentObjectUniqueId=drone_pyb_id, physicsClientId=self.pyb_env.CLIENT)
                else:
                    self.hud_text_id = p.addUserDebugText(hud_text, [0, 0, 0.8], textColorRGB=[0, 0, 0], textSize=1.5, parentObjectUniqueId=drone_pyb_id, replaceItemUniqueId=self.hud_text_id, physicsClientId=self.pyb_env.CLIENT)
                
                # 视线连线
                p.addUserDebugLine(cur_attacker_pos, cur_evader_pos, [0, 1, 1], 1.5, 1.5 / self.CTRL_FREQ, physicsClientId=self.pyb_env.CLIENT)
                
                # 计算两架飞机的空间中点 (Midpoint)
                mid_pos = (cur_attacker_pos + cur_evader_pos) / 2.0

                # 相机跟随逻辑
                self.cam_pos = self.cam_pos * 0.9 + cur_attacker_pos * 0.1
                attacker_rpy = self.pyb_env._getDroneStateVector(attacker_id)[7:10]
                smooth_yaw = np.degrees(attacker_rpy[2]) # 取 Yaw 角

                # ================= 重构的大尺度运镜模式 =================
                if self.camera_mode == 1:
                    # Mode 1: 经典第三人称尾随视角 (类似皇牌空战)
                    # 距离从 2.0 米拉大到 100.0 米，以容纳战斗机的庞大机身和高速运动
                    p.resetDebugVisualizerCamera(100.0, smooth_yaw - 90, -10, self.cam_pos, physicsClientId=self.pyb_env.CLIENT)
                
                elif self.camera_mode == 2:
                    # Mode 2: 战术俯视地图 (Top-down Tactical Map)
                    # 追踪两机中心点，相机高度(距离)动态适应两机的相对距离
                    tactical_dist = max(4000.0, dist_cam * 1.5) 
                    p.resetDebugVisualizerCamera(tactical_dist, 0, -89.9, mid_pos, physicsClientId=self.pyb_env.CLIENT)
                
                elif self.camera_mode == 3:
                    # Mode 3: 动态狗斗视角 (Dynamic Dogfight / Over-the-shoulder)
                    # 相机盯着两机的中心点，但视角方向试图把目标机和主机都纳入画面
                    view_yaw = np.degrees(np.arctan2(cur_evader_pos[1] - cur_attacker_pos[1], cur_evader_pos[0] - cur_attacker_pos[0]))
                    dynamic_dist = np.clip(dist_cam * 1.2, 150.0, 4000.0) # 限制最近和最远距离
                    p.resetDebugVisualizerCamera(dynamic_dist, view_yaw - 90, -15, mid_pos, physicsClientId=self.pyb_env.CLIENT)
                
                elif self.camera_mode == 4:
                    # Mode 4: 目标锁定抵近视角 (Target Tracking)
                    # 镜头死死锁住目标机，距离设为 200 米，方便看清它怎么做机动规避
                    p.resetDebugVisualizerCamera(200.0, drone_angle + 45, -20, cur_evader_pos, physicsClientId=self.pyb_env.CLIENT)
                
                elif self.camera_mode == 5:
                    # Mode 5: 全局大尺度远景
                    # 强制锁定在极高的高空，纵览整个交战空域 (8km x 8km)
                    p.resetDebugVisualizerCamera(10000.0, 45, -30, [0, 0, 3000.0], physicsClientId=self.pyb_env.CLIENT)

            # 在微小帧内，重新计算战术几何 (ATA, AA, HCA)
            # 1. 从当前帧的状态中提取双方的真实物理四元数
            attacker_quat = new_attacker_state[3:7]  
            evader_quat = new_evader_state[3:7]

            # 2. 将四元数转换为旋转矩阵
            rot_mat_A = p.getMatrixFromQuaternion(attacker_quat)
            rot_mat_E = p.getMatrixFromQuaternion(evader_quat)

            # 3. 提取双方的 3D 机头指向向量 (即旋转矩阵的 X 轴正方向)
            # PyBullet 的旋转矩阵是一维数组，X轴对应索引 [0, 3, 6]
            rot_mat_A = p.getMatrixFromQuaternion(attacker_quat)
            rot_mat_E = p.getMatrixFromQuaternion(evader_quat)
            forward_vec_A = np.array([rot_mat_A[0], rot_mat_A[3], rot_mat_A[6]])
            forward_vec_E = np.array([rot_mat_E[0], rot_mat_E[3], rot_mat_E[6]])
            
            # 4. 计算视线向量 (Line of Sight, LOS) 并归一化为单位向量
            # 从攻击机指向目标机的向量
            los_dir = (new_evader_pos - new_attacker_pos) / (new_dist + 1e-6)

            # 5. 利用向量点乘 (Dot Product) 计算三大战术夹角的余弦值 (Cosine)
            # 余弦值范围 [-1, 1]。1 表示方向完全一致，-1 表示方向完全相反。
            
            # 【ATA (攻击机天线偏角)】：攻击机机头 vs 视线方向
            # = 1 时，完美瞄准敌机
            cos_ata_attacker = np.clip(np.dot(forward_vec_A, los_dir), -1.0, 1.0)
            
            # 【AA (攻击机方位角)】：目标机机头 vs 视线方向
            # = 1 时，代表目标机的机头和视线同向，说明攻击机正处于目标机的完美正后方(6点钟)
            cos_aa_attacker = np.clip(np.dot(forward_vec_E, los_dir), -1.0, 1.0)
            
            # 【HCA (航向交叉角)】：攻击机机头 vs 目标机机头
            # = 1 时同向伴飞，= -1 时迎头对冲，接近 0 时是呈十字交叉的剪刀机动
            cos_hca = np.clip(np.dot(forward_vec_A, forward_vec_E), -1.0, 1.0)

            # 兼容保留：将 cos 值转回弧度 
            ata_angle_attacker = np.arccos(cos_ata_attacker)

            # [角色 1] 攻击机 (Attacker) 奖励结算
            if "attacker_0" in actions and not terminations["attacker_0"]:
                TERMINAL_RADIUS = 500.0  # 定义末端冲刺阶段的判定半径

                # 计算双方的高度差 (Z轴距离)
                dz = new_attacker_pos[2] - new_evader_pos[2]

                # 1. 靠近奖励 (全局生效：缩短距离加分，被拉开扣分)
                reward_A_progress = -micro_delta_dist * 20.0 
                reward_A_progress = np.clip(reward_A_progress, -100.0, 100.0)
                reward_A_distance_penalty = - (new_dist / 1000.0) * 1.5 * dt  # 绝对距离势能惩罚

                # 2. 时间惩罚 (全局生效：逼迫速战速决)
                reward_A_time = -0.1 * dt

                # Z 轴共面对齐惩罚 
                reward_A_z_penalty = 0.0
                if abs(dz) > 100.0: # 如果高度差大于 100 米，则开始施加基于高度差的持续惩罚，逼迫主机下降到与目标机大致平齐的高度。
                    reward_A_z_penalty = -(abs(dz) - 100.0) * 0.01 * dt

                # 攻击机软地板警告 
                # 设定 3.0 米为“近地警告线”。低于此高度，每掉 0.1 米扣分越狠
                reward_A_ground_warning = 0.0
                if new_attacker_pos[2] < 500.0:  # 设定 500 米为“近地警告线”。低于此高度，每掉 0.1 米扣分越狠
                    reward_A_ground_warning = -(500.0 - new_attacker_pos[2]) * 0.05 * dt

                reward_A_tracking = 0.0
                reward_A_ramming = 0.0

                if new_dist <= TERMINAL_RADIUS:
                    # --- 末端冲刺阶段 (Terminal Phase) ---
                    # 1. 取消 ATA 瞄准惩罚，彻底释放机动限制
                    reward_A_tracking = 0.0 
                    
                    # 2. 动能冲刺奖励 (Ramming Bonus)
                    # 提取主机当前的绝对线速度，速度越快得分越高，鼓励“踩油门”撞击
                    attacker_vel = self.pyb_env._getDroneStateVector(attacker_id)[10:13]
                    vel_norm = np.linalg.norm(attacker_vel)
                    
                    # ================= 修改：引入水平冲刺系数 =================
                    # 避免主机在最后一刻从天顶垂直“砸”向目标。
                    # 只有当高度差极小时，才给予 100% 的速度冲刺奖励。
                    # 高度差越大，冲刺奖励的折扣越狠。
                    z_alignment_factor = np.clip(200 - abs(dz), 0.0, 1.0)
                    reward_A_ramming = vel_norm * 5.0 * dt * z_alignment_factor
                    # ========================================================
                else:
                    # --- 中程追踪阶段 (Mid-course Phase) ---
    
                    # 只有当攻击机在敌机后半球 (cos_aa_attacker > 0)，且机头大致指向敌机 (cos_ata_attacker > 0) 时，才给予阵位奖励。
                    if cos_ata_attacker > 0 and cos_aa_attacker > 0:
                        # 组合奖励：越接近完美的尾随瞄准 (两者皆趋近于 1)，得分呈指数级上升
                        advantage_score = (cos_ata_attacker * cos_aa_attacker) ** 2
                        reward_A_tracking = advantage_score * 0.5 * dt
                    else:
                        # 如果不在优势阵位，给予轻微的“脱靶惩罚”，逼迫其进行机动
                        # 惩罚力度与机头偏离程度成正比
                        reward_A_tracking = -(1.0 - cos_ata_attacker) * 2.0 * dt
                
                # 单帧结算
                total_rewards["attacker_0"] += (reward_A_progress + reward_A_tracking + reward_A_time + reward_A_ramming + reward_A_z_penalty + reward_A_ground_warning)

            # [角色 2] 目标机 (Evader) 奖励结算
            if "evader_0" in actions and not terminations["evader_0"]:
                WARNING_RADIUS = 300.0  # 告警半径设置
                
                reward_E_escape = 0.0
                reward_E_jinking = 0.0
                reward_E_straight = 0.0
                
                # 苟活奖励 (始终存在)
                reward_E_survival = 0.1 * dt
                
                rel_pos_xy = new_evader_pos[0:2] - new_attacker_pos[0:2]
                dist_xy = np.linalg.norm(rel_pos_xy)

                if dist_xy <= WARNING_RADIUS: # 使用水平距离判断是否触发告警
                    # --- 危险区域：激活逃逸与规避 ---
                    # 1. 逃逸奖励
                    reward_E_escape = micro_delta_dist * 15.0  
                    
                    # 2. 角度破坏奖励 (Spoofing Reward)
                    threat_penalty = 0.0
                    if cos_ata_attacker > 0.5: # 敌机大致看向我 (夹角 < 60度)
                        threat_penalty = - (cos_ata_attacker ** 2) * 2.0 * dt
                    
                    # 奖励项：鼓励诱导敌方进入大 HCA (航向交叉) 的剪刀机动状态
                    # 如果双方在近距离呈大角度交叉 (cos_hca 接近 0 或负数)，说明规避有效
                    hca_reward = 0.0
                    if cos_hca < 0.2: # 航向差异明显，非同向伴飞
                        hca_reward = (0.2 - cos_hca) * 1.5 * dt
                        
                    reward_E_jinking = threat_penalty + hca_reward
                else:
                    # --- 安全区域：鼓励直线平飞 ---
                    evader_action = int(actions["evader_0"])
                    # BFM 动作库中：0=匀速直飞, 1=加速直飞, 2=减速直飞
                    if evader_action in [0, 1, 2]: 
                        reward_E_straight = 0.5 * dt
                        current_z = new_evader_pos[2]
                        # 设定一个期望的安全平飞高度
                        target_altitude = self.evader_initial_z 
                        height_error = abs(target_altitude - current_z)
                        if height_error > 100.0: 
                            # 掉得越多，惩罚越重
                            reward_E_straight -= (height_error * 0.01 * dt)
                    else:
                        # 如果在安全距离做大过载机动，给予轻微惩罚
                        reward_E_straight = -0.3 * dt

                # ================= 新增：目标机软地板警告 =================
                reward_E_ground_warning = 0.0
                if new_evader_pos[2] < 300.0:
                    reward_E_ground_warning = -(300.0 - new_evader_pos[2]) * 0.5 * dt
                # ========================================================
                        
                # 单帧结算
                total_rewards["evader_0"] += (reward_E_escape + reward_E_survival + reward_E_jinking + reward_E_straight + reward_E_ground_warning)

            # 1. 动能撞击 / 击杀成功
            if new_dist < 50.0 and self.macro_step > 2: # 增加暖机帧保护
                if not terminations["attacker_0"]: total_rewards["attacker_0"] += 5000.0
                if not terminations["evader_0"]: total_rewards["evader_0"] -= 5000.0
                terminations["attacker_0"] = True
                terminations["evader_0"] = True

                # 记录终端坐标 (放入 info 字典，供未来测试脚本绘图使用)
                infos["attacker_0"]["terminal_drone_pos"] = new_attacker_state[0:3].copy()
                infos["attacker_0"]["terminal_target_pos"] = new_evader_state[0:3].copy()
                infos["attacker_0"]["reason"] = "success"
                break # 直接结束本轮 AI 决策的 repeat 循环

            # 2. 擦肩而过，触发近炸引信
            # new_dist 是物理步进后的距离，dist 是步进前的距离。
            # 如果进入杀伤圈，且距离开始拉大 (new_dist - dist > 0)，说明刚刚掠过极小值点
            elif new_dist < self.cpa_radius and (new_dist - current_micro_dist) > 0:
                miss_distance = current_micro_dist # 取上一微小帧的极小值
                
                # 根据脱靶量计算梯度得分：基础分1000 + 4000 * (1 - (脱靶量 - 50.0) / 杀伤区间)
                score_ratio = 1.0 - ((miss_distance - 50.0) / (self.cpa_radius - 50.0))
                reward_terminal = 1000.0 + 4000.0 * score_ratio
                
                # 双方进行分数结算 (零和博弈)
                if "attacker_0" in total_rewards and not terminations["attacker_0"]: total_rewards["attacker_0"] += reward_terminal
                if "evader_0" in total_rewards and not terminations["evader_0"]: total_rewards["evader_0"] -= reward_terminal
                
                terminations["attacker_0"] = True
                terminations["evader_0"] = True
                
                # 记录终端坐标
                infos["attacker_0"]["terminal_drone_pos"] = new_attacker_state[0:3].copy()
                infos["attacker_0"]["terminal_target_pos"] = new_evader_state[0:3].copy()
                infos["attacker_0"]["reason"] = "success"
                break

            # 3. 地板/天空边界惩罚
            crash_occurred = False # 新增一个标志位
            for agent, state in zip(["attacker_0", "evader_0"], [new_attacker_state, new_evader_state]):
                if agent in actions and not terminations[agent]: # 只有这个 agent 还在计分板上，才对它进行边界惩罚！
                    if state[2] < 10:
                        total_rewards[agent] -= 100.0
                        terminations[agent] = True
                        infos[agent]["reason"] = "ground_crash"
                        crash_occurred = True 
                    elif state[2] > 15000.0:
                        total_rewards[agent] -= 100.0
                        terminations[agent] = True
                        infos[agent]["reason"] = "out_of_bounds" 
                        crash_occurred = True

            # 只要有飞机坠毁，立刻跳出微观物理循环
            if crash_occurred:
                break

        # --- 退出 Frame Skip 循环，结算当前决策步的最终结果 ---
        
        # 判断是否超时 (Truncation)
        if (self.step_counter / self.CTRL_FREQ) > self.EPISODE_LEN_SEC:
            for agent in self.agents:
                truncations[agent] = True

            # 如果演习结束，且攻击机既没有坠毁也没有击杀（即苟活到了最后），给予巨额惩罚
            if not terminations.get("attacker_0", True) and "attacker_0" in total_rewards:
                total_rewards["attacker_0"] -= 3000.0
                
            # 对应的，目标机成功拖延时间活到了最后，任务圆满完成，给予巨额奖励
            if not terminations.get("evader_0", True) and "evader_0" in total_rewards:
                total_rewards["evader_0"] += 3000.0
        
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

        # --- 计算全局结束标志 ---
        # 在 1v1 中，只要有任何一方死亡或超时，整局对抗立刻结束
        terminations["__all__"] = any(terminations.values()) if terminations else True
        truncations["__all__"] = any(truncations.values()) if truncations else True

        return observations, total_rewards, terminations, truncations, infos
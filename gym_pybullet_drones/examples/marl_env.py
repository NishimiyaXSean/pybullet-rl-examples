import numpy as np
import gymnasium as gym
import pybullet as p
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import time

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from tacview_logger import TacviewLogger

class Drone1v1MARLEnv(MultiAgentEnv):
    metadata = {"render_modes": ["human"], "name": "drone_1v1_v0"}

    def __init__(self, gui=False, record_tacview=False):
        super().__init__()
        # 1. 明确定义智能体身份 (PettingZoo 规范核心)
        self.possible_agents = ["attacker_0", "evader_0"]
        # 在运行中，如果某架飞机坠毁，它会从 self.agents 列表中被移除
        self.agents = self.possible_agents[:]

        # 2. 实例化底层物理引擎 (CtrlAviary)
        # 将两架飞机分别放置在场地的对角线位置，拉开初始距离
        init_xyzs = np.array([
            [-500.0, 0.0, 3000.0],  # attacker_0 的初始位置 (ID: 0)
            [ 500.0, 0.0, 3000.0]   # evader_0   的初始位置 (ID: 1)
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
        self.EPISODE_LEN_SEC = 240 # 回合最大时长
        self.cpa_radius = 300.0   # 近炸引信触发半径

        # --- 战斗机飞行包线参数 (F-16/歼-10 级别模拟) ---
        self.MAX_G = 9.0          # 最大结构过载 (正G)
        self.MIN_G = -3.0         # 最大负过载 (通常远小于正G)
        self.CORNER_SPEED = 150.0 # 角速度 (约 540 km/h)：能拉出最大过载的最低速度
        self.MAX_SPEED = 400.0    # 绝对最大平飞速度 (约 1.2 马赫)
        self.STALL_SPEED = 60.0   # 基础失速速度
        self.g = 9.81             # 重力加速度
        
        # 动作空间：离散的 13 种 BFM 动作
        self.action_spaces = {
            agent: gym.spaces.Discrete(13)
            for agent in self.possible_agents
        }

        # 植入你的 BFM 映射字典
        self.bfm_action_mapping = {
            0:  ( 0,  1,  0.0),            # a1: 匀速直飞
            1:  ( 2,  1,  0.0),            # a2: 加速直飞
            2:  (-2,  1,  0.0),            # a3: 减速直飞
            3:  ( 0,  8,  0.0),            # a4: 暴力跃升 (8G)
            4:  ( 0, -8,  0.0),            # a5: 暴力俯冲 (-8G)
            5:  ( 0,  8,  np.pi / 3.0),    # a6: 左转跃升
            6:  ( 0, -8, -np.pi / 3.0),    # a7: 右转俯冲
            7:  ( 0,  8, -np.pi / 3.0),    # a8: 右转跃升
            8:  ( 0, -8,  np.pi / 3.0),    # a9: 左转俯冲
            9:  ( 0,  2, -np.pi / 3.0),    # a10: 缓和右转 (2G)
            10: ( 0,  2,  np.pi / 3.0),     # a11: 缓和左转 (2G)

            # === 新增：用于末端垂直微调的缓和机动 ===
            11: ( 0,  3,  0.0),            # a12: 缓和跃升 (3G)
            12: ( 0, -3,  0.0),            # a13: 缓和俯冲 (-3G)
        }
        
        # MAPPO 专属 Dict 观测空间
        # 假设全局状态包含2架飞机的绝对物理参数：位置(3)+四元数(4)+线速度(3)+角速度(3) = 13维/架
        # 1v1 的总全局维度为 26。未来如果是 2v2，这里相应增加即可。
        self.GLOBAL_STATE_DIM = 26 
        
        self.observation_spaces = {
            agent: gym.spaces.Dict({
                "obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32),
                "global_state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.GLOBAL_STATE_DIM,), dtype=np.float32)
            })
            for agent in self.possible_agents
        }

        # 视觉与运镜初始化
        self.camera_mode = 1  # 默认相机视角 (1:智能追尾)
        self.hud_text_id = -1
        self.fuze_obj_id = -1
        self.last_draw_pos = np.zeros(3)
        self.last_target_draw_pos = np.zeros(3)
        self.cam_pos = np.zeros(3)

        # 初始化 Tacview 记录器
        self.record_tacview = record_tacview
        if self.record_tacview:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_filename = f"drone_eval_{timestamp}.txt.acmi"
            self.tacview_logger = TacviewLogger(filename=log_filename)

        self.curriculum_stage = 1
        self._update_curriculum_bounds()

    def set_curriculum_stage(self, stage):
        """供外部 RLlib 算法调用的难度调节接口"""
        self.curriculum_stage = stage
        self._update_curriculum_bounds()

    def _update_curriculum_bounds(self):
        """定义每个难度阶段的具体出生范围"""
        if self.curriculum_stage == 1:
            # Stage 1: 新手村 (限制极大的固定靶/呆板靶)
            self.d_min, self.d_max = 600.0, 1000.0
            self.z_min, self.z_max = 1500.0, 2000.0
            self.EVADER_SPEED_COEFF = 0.50     # 约 0.6 马赫
            self.EVADER_G_COEFF = 0.333        # 约 3.0 G
            self.warning_radius = 1500.0       # 极小的告警圈
            
        elif self.curriculum_stage == 2:
            # Stage 2: 中距拉锯 (解禁部分机动能力与战术视野)
            self.d_min, self.d_max = 1000.0, 1500.0
            self.z_min, self.z_max = 1800.0, 2500.0
            self.EVADER_SPEED_COEFF = 0.65     # 提速至约 0.8 马赫
            self.EVADER_G_COEFF = 0.55         # 解禁至约 5.0 G (允许中等烈度规避)
            self.warning_radius = 3000.0       # 告警半径翻倍，提前开启规避动作
            
        else:
            # Stage 3: 长程高空对决 (毕业期：近乎全对称的生死斗)
            self.d_min, self.d_max = 1500.0, 2500.0
            self.z_min, self.z_max = 2200.0, 3200.0
            self.EVADER_SPEED_COEFF = 0.85     # 极其接近主机的速度
            self.EVADER_G_COEFF = 0.85         # 极其接近主机的机动性
            self.warning_radius = 10000.0      # 相当于全图告警，开局即处于博弈状态
    def _compute_global_state(self):
        """
        为 MAPPO 的 Critic 提取全知全能的全局状态 (Global State)
        """
        global_state = []
        
        # 遍历所有可能存在的飞机 (注意使用 self.possible_agents 保证顺序和维度固定)
        for i, agent in enumerate(self.possible_agents):
            if agent in self.agents:
                # _getDroneStateVector 必须传入内部索引 (0 或是 1)，不能传 PyBullet 实体 ID
                state_vec = self.pyb_env._getDroneStateVector(i)
                
                # 提取: 绝对位置(3), 四元数姿态(4), 绝对速度(3), 绝对角速度(3)
                pos = state_vec[0:3] / 5000.0          # 归一化位置
                quat = state_vec[3:7]                  # 四元数本身就在 [-1, 1]
                vel = state_vec[10:13] / self.MAX_SPEED # 归一化速度
                ang_vel = state_vec[13:16] / np.pi      # 归一化角速度
                
                agent_state = np.concatenate([pos, quat, vel, ang_vel])
            else:
                # 填充死亡零向量 (Padding)
                # 在 N vs M 中，如果有飞机被击落，必须用全 0 占位以保证神经网络输入维度不变
                agent_state = np.zeros(13, dtype=np.float32)
                
            global_state.append(agent_state)
            
        # 拼接成一个展平的一维大向量
        global_array = np.concatenate(global_state).astype(np.float32)
        return np.clip(global_array, -1.0, 1.0)

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

        # 2. 在该象限内，生成初始距离
        attacker_x = sign_x * np.random.uniform(self.d_min, self.d_max)
        attacker_y = sign_y * np.random.uniform(self.d_min, self.d_max)
        attacker_z = np.random.uniform(self.z_min, self.z_max)

        # 3. 目标机强制取相反符号，确保永远出生在对角象限！
        evader_x = -sign_x * np.random.uniform(self.d_min, self.d_max)
        evader_y = -sign_y * np.random.uniform(self.d_min, self.d_max)

        # 让目标机的高度以攻击机为基准，上下随机浮动 500 米
        # 这样攻击机有 50% 概率处于高位，50% 概率处于低位，必须学会全向俯仰机动！
        raw_evader_z = attacker_z + np.random.uniform(-500.0, 500.0)

        # 【修改】：给目标机更多的向下生成空间
        evader_z = np.clip(raw_evader_z, 1000.0, 3600.0) 
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

        # 动态获取当前 Stage 下目标机被允许的极限速度
        evader_max_speed = self.MAX_SPEED * self.EVADER_SPEED_COEFF

        # 替换 reset 函数中原本的初始姿态和速度赋值：
        for i, agent in enumerate(self.agents):
            initial_pos = new_init_xyzs[i]
            pyb_id = self.pyb_env.DRONE_IDS[i] if hasattr(self.pyb_env, 'DRONE_IDS') else self.pyb_env.drone_ids[i]
            
            # 计算指向原点 (0,0) 的基础偏航角
            dx = -initial_pos[0]
            dy = -initial_pos[1]
            base_yaw = np.arctan2(dy, dx)
            
            if agent == "attacker_0":
                yaw = base_yaw
                self.attacker_init_yaw = yaw      # 记录攻击机朝向基准
                current_init_speed = 150.0        # 攻击机初始速度维持 150 m/s
            else:         
                # ================= 课程学习：全向战术拦截态势 =================
                # 0: 纯尾追 | ±np.pi/2: 侧向交叉 | np.pi: 迎头对冲
                tactical_offset = np.random.choice([0.0, np.pi/2, -np.pi/2, np.pi])
                
                # 添加随机扰动 (约 ±15 度)，防止网络死记硬背
                noise = np.random.uniform(-np.pi/12, np.pi/12)
                yaw = self.attacker_init_yaw + tactical_offset + noise
                
                # 【关键修复】目标机不再硬编码 250m/s，而是使用当前阶段的极速
                current_init_speed = evader_max_speed 
                # ================================================================
            
            # 【关键修复】使用 current_init_speed 而不是 initial_speed
            init_vel = [current_init_speed * np.cos(yaw), current_init_speed * np.sin(yaw), 0.0]
            init_quat = p.getQuaternionFromEuler([0, 0, yaw])
            
            p.resetBasePositionAndOrientation(pyb_id, initial_pos, init_quat, physicsClientId=self.pyb_env.CLIENT)
            p.resetBaseVelocity(pyb_id, linearVelocity=init_vel, physicsClientId=self.pyb_env.CLIENT)

        if hasattr(self.pyb_env, '_updateAndStoreKinematicInformation'):
            self.pyb_env._updateAndStoreKinematicInformation()
        
        # 初始化时间步与两架飞机的局部追踪变量
        self.step_counter = 0  # 留着给底层备用
        self.macro_step = 0    # 真正的宏观决策步数

        # 计算开局时的初始距离 (用于第一帧的奖励计算基准)
        attacker_pos = self.pyb_env._getDroneStateVector(0)[0:3]
        evader_pos = self.pyb_env._getDroneStateVector(1)[0:3]
        self.prev_dist = np.linalg.norm(attacker_pos - evader_pos)

        # 记录上一帧的 ATA 余弦值，用于计算趋势
        rot_mat_init_A = p.getMatrixFromQuaternion(self.pyb_env._getDroneStateVector(0)[3:7])
        forward_init_A = np.array([rot_mat_init_A[0], rot_mat_init_A[3], rot_mat_init_A[6]])
        los_init_dir = (evader_pos - attacker_pos) / (self.prev_dist + 1e-6)
        self.last_cos_ata_A = np.clip(np.dot(forward_init_A, los_init_dir), -1.0, 1.0)

        '''
        # ================= 课程学习 Stage 2：随机化目标机盘旋 =================
        # 随机决定本回合目标机的机动策略。
        self.evader_maneuver = np.random.choice(
            ["straight", "turn_left", "turn_right"], 
            # p=[1.0, 0.0, 0.0]  # 目前 100% 直飞
            p=[0.9, 0.05, 0.05]  # 概率分布：90% 直飞，5% 左转，5% 右转
        )
        # ====================================================================
        '''
    
        global_state_array = self._compute_global_state()
        obs_dict = {
            agent: {
                "obs": self._compute_obs(agent),
                "global_state": global_state_array
            }
            for agent in self.agents
        }
        
        info_dict = {agent: {} for agent in self.agents}
        
        # 3D 视觉场景构建 (仅在开启 GUI 时生效)
        self.last_draw_pos = attacker_pos.copy()
        self.last_target_draw_pos = evader_pos.copy()
        self.cam_pos = attacker_pos.copy()

        if self.pyb_env.GUI:
            p.removeAllUserDebugItems(physicsClientId=self.pyb_env.CLIENT) # 清理上一局的残留线条
            
            # 高空战术参考网格
            # 设定在 1500米、3000米、4500米 绘制三层不同颜色的半透明网格
            grid_altitudes = {
                1500.0: [0.0, 0.5, 1.0],  # 浅蓝色 (低空参考线)
                3000.0: [0.0, 1.0, 0.5],  # 青绿色 (中空参考线)
                4500.0: [1.0, 0.5, 0.0]   # 橙色   (高空警告线)
            }
            
            grid_size = 4000.0 # 缩小网格覆盖范围到正负 4km (8km x 8km)
            major_step = 1000.0 # 主刻度：每 1000m 一条粗线 (大局观)
            minor_step = 250.0  # 次刻度：每 250m 一条细线 (精细速度感)
            
            for z, color in grid_altitudes.items():
                # 1. 绘制次级网格 (细线)
                for y in np.arange(-grid_size, grid_size + 1, minor_step):
                    p.addUserDebugLine([-grid_size, y, z], [grid_size, y, z], color, 0.5, 0, physicsClientId=self.pyb_env.CLIENT)
                for x in np.arange(-grid_size, grid_size + 1, minor_step):
                    p.addUserDebugLine([x, -grid_size, z], [x, grid_size, z], color, 0.5, 0, physicsClientId=self.pyb_env.CLIENT)
                    
                # 2. 绘制主级网格 (粗线覆盖)
                for y in np.arange(-grid_size, grid_size + 1, major_step):
                    p.addUserDebugLine([-grid_size, y, z], [grid_size, y, z], color, 2.0, 0, physicsClientId=self.pyb_env.CLIENT)
                for x in np.arange(-grid_size, grid_size + 1, major_step):
                    p.addUserDebugLine([x, -grid_size, z], [x, grid_size, z], color, 2.0, 0, physicsClientId=self.pyb_env.CLIENT)
                    
            # 为目标机生成一个半透明的近炸引信杀伤圈
            fuze_v_id = p.createVisualShape(p.GEOM_SPHERE, radius=self.cpa_radius, rgbaColor=[1, 0.5, 0, 0.25])
            self.fuze_obj_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=fuze_v_id, basePosition=evader_pos, physicsClientId=self.pyb_env.CLIENT)

            z_offset = 0.05
            # 绘制主坐标轴
            p.addUserDebugLine([-15, 0, z_offset],[15, 0, z_offset], [1, 0, 0], 4, 0, physicsClientId=self.pyb_env.CLIENT)
            p.addUserDebugLine([0, -15, z_offset], [0, 15, z_offset],[0, 1, 0], 4, 0, physicsClientId=self.pyb_env.CLIENT)
            p.addUserDebugLine([0, 0, z_offset],[0, 0, 15], [0, 0.5, 1], 4, 0, physicsClientId=self.pyb_env.CLIENT)
            
            # ================= 战术雷达标记 (巨型幽灵球) =================
            # 创造半径 150 米的巨大球体。只赋予 Visual Shape，不赋予 Collision Shape
            # 这样它们完全没有物理碰撞，不会干扰强化学习的动力学环境
            v_shape_A = p.createVisualShape(p.GEOM_SPHERE, radius=150.0, rgbaColor=[1.0, 0.2, 0.2, 0.7]) # 红色主机
            v_shape_E = p.createVisualShape(p.GEOM_SPHERE, radius=150.0, rgbaColor=[1.0, 0.8, 0.0, 0.7]) # 橙黄色目标
            
            # 使用 baseMass=0 且无碰撞体的方式创建实体
            self.radar_marker_A = p.createMultiBody(baseMass=0, baseVisualShapeIndex=v_shape_A, basePosition=attacker_pos, physicsClientId=self.pyb_env.CLIENT)
            self.radar_marker_E = p.createMultiBody(baseMass=0, baseVisualShapeIndex=v_shape_E, basePosition=evader_pos, physicsClientId=self.pyb_env.CLIENT)
            # ==========================================================

        # 在回合开始时重置并启动录制
        if getattr(self, 'record_tacview', False):
            # 如果上一次没关干净，先关掉
            if hasattr(self, 'tacview_logger'):
                self.tacview_logger.close()
            
            # 开启新一轮录制
            self.tacview_logger.start()
            
            # 记录第 0 帧 (初始状态)
            self._record_tacview_frame(time_sec=0.0)

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

        # 3. 核心坐标转换：构建真正的体轴坐标系 (Body Frame)
        # 直接使用底层物理引擎给出的完整 3D 姿态四元数
        my_full_quat = my_state[3:7] 
        
        # 获取从世界坐标系到当前飞机体轴坐标系的逆变换矩阵
        _, inv_quat = p.invertTransform([0, 0, 0], my_full_quat)

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
        MAX_DIST = 8000.0     
        MAX_HEIGHT = 5000.0
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
        AI_DECISION_DT = 0.5  # 强制规定 AI 每 0.5 秒做一次决策 (在 60Hz 的底层频率下，相当于推进 30 帧)
        dynamic_frame_skip = int(AI_DECISION_DT * self.CTRL_FREQ)
        dt = 1 / self.CTRL_FREQ

        total_rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        infos = {agent: {} for agent in self.agents}

        # 提取两架飞机的初始状态 (用于后续计算奖励和碰撞)
        attacker_id = 0
        evader_id = 1
        attacker_state_init = self.pyb_env._getDroneStateVector(attacker_id)
        evader_state_init = self.pyb_env._getDroneStateVector(evader_id)
        attacker_rpy = attacker_state_init[7:10]
        evader_rpy = evader_state_init[7:10]

        dist = np.linalg.norm(attacker_state_init[0:3] - evader_state_init[0:3])
        current_micro_dist = dist

        self.macro_step += 1 # 新增：每次 AI 下达指令，宏观步数推进 1 步

        '''
        # ================= 进阶 Stage 3.5：启发式 RWR 与高频 3D 机动 =================
        # 1. 雷达告警接收机 (RWR) 紧急规避逻辑
        # 如果攻击机逼近到 1500 米内，且机头正对目标机 (ATA < 30度，即 cos_ata > 0.866)
        if current_micro_dist < 1500.0 and getattr(self, 'last_cos_ata_A', 0.0) > 0.866:
            # 根据当前高度决定逃逸策略
            if trusted_states["evader_0"]["pos"][2] > 2500.0:
                # 高空被锁定：执行极其难缠的 3D 螺旋俯冲逃逸 (极速掉高 + 转向)
                self.evader_maneuver = "spiral_dive"
            else:
                # 低空/中空被锁定：执行最大过载的水平急转弯 (Break Turn)
                self.evader_maneuver = "hard_turn_left" if np.random.rand() > 0.5 else "hard_turn_right"
        

        if np.random.rand() < 0.15:  # 期望值每 3.3 秒 (6.6个宏观步) 重新评估一次战术
            self.evader_maneuver = np.random.choice(
                ["straight", "turn_left", "turn_right", "climb", "dive"], 
                # 概率分布：大幅减少呆板的直飞，加入爬升和俯冲
                p=[0.3, 0.25, 0.25, 0.1, 0.1]
            )
        '''
        # 绝对信任的本地物理账本
        trusted_states = {
            "attacker_0": {
                "pos": attacker_state_init[0:3].copy(), 
                "vel": attacker_state_init[10:13].copy(),
                "mu": attacker_rpy[0]  # 提取初始滚转角 (Roll)
            },
            "evader_0":   {
                "pos": evader_state_init[0:3].copy(),   
                "vel": evader_state_init[10:13].copy(),
                "mu": evader_rpy[0]
            }
        }

        # 物理控制量的一阶惯性缓冲池
        # 用于记录飞机当前真实的过载状态，防止 0.5s 宏观决策造成的瞬间受力突变
        current_controls = {
            "attacker_0": {"n_x": 0.0, "n_n": 1.0}, # 初始假定为匀速直飞 (1G)
            "evader_0":   {"n_x": 0.0, "n_n": 1.0}
        }

        for _ in range(dynamic_frame_skip):           
            self.pyb_env.step(np.zeros((2, 4)))
            self.step_counter += 1
            
            for i, agent in enumerate(["attacker_0", "evader_0"]):
                if agent not in actions or terminations[agent]: # 如果这架飞机已经判定死亡，则直接跳过
                    continue

                pyb_id = self.pyb_env.DRONE_IDS[i] if hasattr(self.pyb_env, 'DRONE_IDS') else self.pyb_env.drone_ids[i]
                
                # 1. 提取当前帧的绝对信任坐标、速度与滚转角
                pos = trusted_states[agent]["pos"]
                vel = trusted_states[agent]["vel"]
                current_mu = trusted_states[agent]["mu"]

                # 2. 动态确定当前飞机的结构极限参数
                current_max_g = self.MAX_G
                current_min_g = self.MIN_G
                current_max_speed = self.MAX_SPEED
                
                if agent == "evader_0":
                    current_max_g = self.MAX_G * self.EVADER_G_COEFF
                    current_min_g = self.MIN_G * self.EVADER_G_COEFF
                    current_max_speed = self.MAX_SPEED * self.EVADER_SPEED_COEFF

                # 3. 离散动作解包 (BFM 指令)
                action_idx = int(actions.get(agent, 0))   # 安全解包，哪怕上层没传 evader 的动作，默认给 0 (匀速直飞)
                
                # ================= 新增：目标机硬性动作接管 =================
                # 如果是目标机，且当前距离大于告警半径，强行剥夺 AI 的控制权
                if agent == "evader_0" and current_micro_dist > self.warning_radius:
                    # 强制替换为动作 0 (匀速直飞：1G法向过载，0滚转，0加减速)
                    action_idx = 0
                # ============================================================

                n_x_cmd, n_n_cmd, target_mu = self.bfm_action_mapping[action_idx]

                '''
                # 加入水平盘旋动作 
                if agent == "evader_0":
                    if self.evader_maneuver == "turn_left":
                        n_x_cmd = 0.0          # 保持匀速
                        n_n_cmd = 2.0          # 2G 法向过载
                        target_mu = np.pi / 3.0   # 60度滚转 (保持高度不掉)
                    elif self.evader_maneuver == "turn_right":
                        n_x_cmd = 0.0
                        n_n_cmd = 2.0
                        target_mu = -np.pi / 3.0  # 向右 60度滚转
                    elif self.evader_maneuver == "climb":
                        n_x_cmd = 0.0
                        n_n_cmd = 2.5            # 2.5G 缓和跃升
                        target_mu = 0.0
                    elif self.evader_maneuver == "dive":
                        n_x_cmd = 0.0
                        n_n_cmd = -1.0           # 负 1G 俯冲
                        target_mu = 0.0
                    else: # 直飞
                        n_x_cmd = 0.0
                        n_n_cmd = 1.0
                        target_mu = 0.0
                '''

                # ================= 核心修复：分化 GPWS 触发高度 =================
                # 主机在 300m 极低空拉起，而目标机的死亡线是 595m，必须在 800m 提前拉起
                gpws_trigger_alt = 300.0 if agent == "attacker_0" else 800.0
                
                # GPWS 近地警告最高优先级覆盖
                if pos[2] < gpws_trigger_alt and vel[2] < -5.0:   
                    n_n_cmd = current_max_g  # 强制给足最大过载拉起
                    target_mu = 0.0          # 强制改平
                # ================================================================

                # 4. === 核心：物理平滑过渡机制 (一阶惯性延迟) ===
                # tau_g 是飞机的过载建立时间常数 (秒)。0.4 秒意味着指令下达后，约 0.4 秒达到目标 G 值的 63%
                tau_g = 0.4 
                alpha_g = dt / (tau_g + dt)
                
                # 平滑逼近：真实过载 = 上一帧过载 + (目标过载 - 上一帧过载) * 平滑系数
                current_controls[agent]["n_x"] += (n_x_cmd - current_controls[agent]["n_x"]) * alpha_g
                current_controls[agent]["n_n"] += (n_n_cmd - current_controls[agent]["n_n"]) * alpha_g
                
                actual_n_x = current_controls[agent]["n_x"]
                actual_n_n = current_controls[agent]["n_n"]

                # 滚转角控制 (通过 P 控制器平滑逼近绝对角度)
                roll_error = target_mu - current_mu
                roll_error = (roll_error + np.pi) % (2 * np.pi) - np.pi # 限制在 [-pi, pi]
                
                MAX_ROLL_RATE = np.pi # 滚转率限制：180度/秒
                roll_rate_cmd = np.clip(roll_error * 4.0, -MAX_ROLL_RATE, MAX_ROLL_RATE) # 增益系数为 4.0
                mu_cmd = current_mu + roll_rate_cmd * dt

                # 5. 包线限制计算准备
                V = np.linalg.norm(vel)
                if V < 1e-3: V = 1e-3  # 防止除以 0 导致数值崩溃

                # A. 升力限制 (低速时无法拉出大过载，升力与速度的平方成正比)
                # 物理依据：升力与速度的平方成正比。计算当前速度下能拉出的极限 G 值。
                available_n_lift = ((V / self.CORNER_SPEED) ** 2) * current_max_g
                
                # B. 实际最大可用正负 G 是结构极限与空气动力学极限的交集
                actual_max_n = min(current_max_g, available_n_lift)
                actual_min_n = max(current_min_g, -available_n_lift)
                
                # C. 将网络输出的 G 值强制压入真实的 V-n 包线内
                n_n = np.clip(actual_n_n, actual_min_n, actual_max_n)
                mu = mu_cmd

                # D. 切向过载 (加减速) 的简易动力学限制
                n_x = actual_n_x
                if V > current_max_speed and actual_n_x > 0:
                    n_x = 0.0  # 超过极速无法继续加速 (阻力壁垒)
                elif V < self.STALL_SPEED and actual_n_x < 0:
                    n_x = 0.0  # 接近失速时无法继续减速

                # 6. 开始欧拉积分计算姿态更新
                gamma = np.arcsin(np.clip(vel[2] / V, -1.0, 1.0)) # 航迹俯仰角
                chi = np.arctan2(vel[1], vel[0])                  # 航迹方位角
                
                V_dot = self.g * (n_x - np.sin(gamma))
                
                # 防止大俯仰角时出现奇点 (gamma 接近 90 度时 cos(gamma) 接近 0)
                cos_gamma = np.cos(gamma) if abs(np.cos(gamma)) > 1e-3 else 1e-3
                
                gamma_dot = (self.g / V) * (n_n * np.cos(mu) - np.cos(gamma))
                chi_dot = (self.g * n_n * np.sin(mu)) / (V * cos_gamma)
                
                # 欧拉积分更新状态
                new_V = V + V_dot * dt
                # 新增防超速与防倒车机制
                new_V = np.clip(new_V, 20.0, current_max_speed)
                new_gamma = gamma + gamma_dot * dt
                new_chi = chi + chi_dot * dt
                
                # 将极坐标下的速度转换回 3D 笛卡尔坐标系
                new_vel = np.array([
                    new_V * np.cos(new_gamma) * np.cos(new_chi),
                    new_V * np.cos(new_gamma) * np.sin(new_chi),
                    new_V * np.sin(new_gamma)
                ])
                
                new_pos = pos + new_vel * dt

                # 【关键修复】：必须在调用底层 API 之前拦截所有 NaN 和 Inf！
                # 检查速度、位置，以及即将用于计算四元数的角度是否正常
                if (not np.all(np.isfinite(new_pos)) or 
                    not np.all(np.isfinite(new_vel)) or 
                    not np.isfinite(mu) or 
                    not np.isfinite(new_gamma) or 
                    not np.isfinite(new_chi)):
                    
                    # 如果发生数值爆炸，强制判定为坠毁安全状态，截断崩溃传播
                    new_pos = np.array([pos[0], pos[1], 0.0]) # 强制拍在海平面
                    new_vel = np.zeros(3)
                    new_quat = p.getQuaternionFromEuler([0, 0, 0])
                else:
                    # 只有在所有浮点数都健康的情况下，才允许计算四元数
                    new_quat = p.getQuaternionFromEuler([mu, new_gamma, new_chi])

                # 高度限制与天花板惩罚 
                if agent == "attacker_0":
                    # 我方无人机：触碰天花板时给予持续惩罚，防止利用边界“滑行”
                    if new_pos[2] > 5000.0 :
                        new_pos[2] = 5000.0
                        total_rewards[agent] -= 0.5 * dt  # 累加高度软惩罚
                    # 正常防钻地（不给惩罚，直接限制）
                    elif new_pos[2] < 1.0:
                        new_pos[2] = 1.0
                else:
                    # ================= 目标机物理边界 (完全交由神经网络控制) =================
                    if new_pos[2] > 4000.0:
                        new_pos[2] = 4000.0
                        total_rewards[agent] -= 0.5 * dt  # 触碰天花板同样给点软惩罚

                # ================= 核心修复：更新本地账本并强制洗白 PyBullet =================
                trusted_states[agent]["pos"] = new_pos
                trusted_states[agent]["vel"] = new_vel
                trusted_states[agent]["mu"] = mu

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

            # 在微小帧内，重新计算战术几何 (ATA, AA, HCA)
            # 1. 从当前帧的状态中提取双方的真实物理四元数
            attacker_quat = new_attacker_state[3:7]  
            evader_quat = new_evader_state[3:7]

            # 2. 将四元数转换为旋转矩阵
            rot_mat_A = p.getMatrixFromQuaternion(attacker_quat)
            rot_mat_E = p.getMatrixFromQuaternion(evader_quat)

            # 3. 提取双方的 3D 机头指向向量 (即旋转矩阵的 X 轴正方向)
            # PyBullet 的旋转矩阵是一维数组，X轴对应索引 [0, 3, 6]
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
                TERMINAL_RADIUS = 400.0  # 定义末端冲刺阶段的判定半径

                # 计算双方的高度差 (Z轴距离)
                dz = new_attacker_pos[2] - new_evader_pos[2]

                # 1. 靠近奖励 
                if micro_delta_dist < 0:
                    reward_A_progress = abs(micro_delta_dist) * 0.1 # 靠近给分
                else:
                    reward_A_progress = -micro_delta_dist * 0.03

                # 2. 时间惩罚 (全局生效：逼迫速战速决)
                time_ratio = (self.step_counter / self.CTRL_FREQ) / self.EPISODE_LEN_SEC
                # 时间惩罚随时间推移越来越重：开局 -0.5 分/秒，快结束时变成 -2.5 分/秒
                reward_A_time = -(0.5 + time_ratio * 2.0) * dt

                reward_A_z_advantage = 0.0
                # 【优化】：必须在“积极进攻（距离正在拉近）”且“机头对准目标（前半球）”时，才给予势能奖励。
                if dz > 50.0 and cos_ata_attacker > 0.5 and micro_delta_dist < 0:
                    # 降低低保权重，逼迫其寻求击杀
                    reward_A_z_advantage = np.clip(dz, 0.0, 1000.0) * 0.001 * dt
                elif dz < -100.0:
                    # 【修复】目标在上方时，大幅增加下方惩罚，逼迫它拉机头爬升
                    penalty_scale = np.clip(abs(dz) / 300.0, 1.0, 4.0) 
                    reward_A_z_advantage -= abs(dz) * 0.02 * penalty_scale * dt
                
                reward_A_energy_loss = 0.0 

                # 【新增防悬停机制】：如果速度跌到谷底（接近失速），严厉惩罚！
                current_v = np.linalg.norm(trusted_states["attacker_0"]["vel"])
                vz_A = trusted_states["attacker_0"]["vel"][2]
                
                # 【核心修复】：免除积极爬升时的低速惩罚！(动能换势能是合理的)
                # 只有当速度极低，且飞机没有在垂直向上爬升时，才算作“危险低速”
                if current_v < 150.0 and vz_A < 10.0:
                    reward_A_energy_loss -= (150.0 - current_v) * 0.5 * dt

                # 攻击机软地板警告 
                reward_A_ground_warning = 0.0
                if new_attacker_pos[2] < 200.0:  
                    # 高度越低，惩罚呈指数级上升
                    depth_ratio = (200.0 - new_attacker_pos[2]) / 200.0
                    reward_A_ground_warning = -(depth_ratio ** 2) * 5.0 * dt

                    # 提取当前 Z 轴速度 (垂直速度)
                    vz = trusted_states["attacker_0"]["vel"][2]
                    
                    # 【核心保命机制】如果处于低空，且还在向下掉高度
                    if vz < -1.0: 
                        # 下坠越快，乘法叠加的惩罚越极端 (动态势能墙)
                        # 例如 vz = -50m/s 时，每秒扣除巨大的分数，逼迫网络产生对“死亡俯冲”的恐惧
                        reward_A_ground_warning -= abs(vz) * 0.2 * dt
                
                # 计算相对速度矢量，用于指引“直线提前量拦截”
                vel_A = trusted_states["attacker_0"]["vel"]
                vel_E = trusted_states["evader_0"]["vel"]
                rel_vel = vel_A - vel_E
                rel_vel_dir = rel_vel / (np.linalg.norm(rel_vel) + 1e-6)
                
                # cos_collision 衡量的是“相对速度”是否指向目标，这是直线拦截的核心！
                cos_collision = np.clip(np.dot(rel_vel_dir, los_dir), -1.0, 1.0)
                
                reward_A_tracking = 0.0

                # 1. 相对速度追踪奖励 (大幅提高权重，引导前置拦截)
                if cos_collision > 0.0:
                    # 距离越近，速度指向的奖励权重越高 (逼迫网络在近距离切内圈)
                    dynamic_collision_weight = 8.0 + (1000.0 / (new_dist + 100.0)) * 5.0
                    reward_A_tracking += cos_collision * dynamic_collision_weight * dt
                else:
                    # 速度背离目标时，轻微惩罚
                    reward_A_tracking += cos_collision * 5.0 * dt

                # 2. ATA 机头指向奖励
                base_ata_reward = 0.0
                if cos_ata_attacker > 0.0:
                    # 机头在前半球，计算基础奖励
                    base_ata_reward = cos_ata_attacker * 4.0 * dt
                    if cos_ata_attacker > 0.866: # 进入前 30 度 (高阶锁定)
                        base_ata_reward += 2.0 * dt
                    
                    # 末端等高约束 (共面惩罚)
                    if new_dist < self.warning_radius:
                        # 【修改】：容差缩小到 100 米，高度误差大于 100 米就开始严厉打折
                        z_error = max(0.0, abs(dz) - 100.0)
                        # 【修改】：一旦高度差超过 500 米，跟踪奖励直接归零 (1.0 惩罚系数)
                        z_penalty_factor = np.clip(z_error / 500.0, 0.0, 1.0)
                        
                        if micro_delta_dist < 0:
                            reward_A_tracking += base_ata_reward * (1.0 - z_penalty_factor)
                    else:                  
                        # 远距离时，不在乎高度差，全额给分
                        if micro_delta_dist < 0:
                            reward_A_tracking += base_ata_reward
                        else:
                            reward_A_tracking += base_ata_reward * 0.1
                else:
                    # 机头在后半球，给予严厉的持续惩罚
                    reward_A_tracking += cos_ata_attacker * 5.0 * dt

                # 3. 阵位优势：适度保留
                if cos_aa_attacker > 0.5: 
                    reward_A_tracking += cos_aa_attacker * 2.0 * dt
                
                # 4. HCA 航向交叉角奖励 - 鼓励同向伴飞，惩罚迎头对冲
                if cos_hca > 0.0:
                    # cos_hca > 0 表示双方夹角小于 90 度 (大致往同一个方向飞)
                    # 给予小额奖励，鼓励航向对齐 (Trajectory Alignment)
                    reward_A_tracking += cos_hca * 1.0 * dt
    
                # 横向侧滑惩罚 (Sideslip Penalty)
                # 惩罚飞机在进行追踪时出现的多余滚转或侧向速度，逼迫模型采用更有效率的纯垂直/水平机动
                # local_vel[1] 是体轴坐标系下的 Y 轴速度 (侧滑速度)
                sideslip_vel = abs(trusted_states["attacker_0"]["vel"][1])
                reward_A_tracking -= (sideslip_vel / self.MAX_SPEED) * 2.0 * dt
        
                reward_A_ramming = 0.0

                # ================= 核心优化：全局接近率 (Closing Speed) 奖励 =================
                # 无论多远，只要速度矢量指向目标（拉近距离），就给予奖励，指引它长程极速冲刺！
                los_vec = new_evader_pos - new_attacker_pos
                los_dir = los_vec / (np.linalg.norm(los_vec) + 1e-6)
                attacker_vel = trusted_states["attacker_0"]["vel"]
                
                # 接近率 = 速度在视线方向上的投影
                closing_speed = np.dot(attacker_vel, los_dir) 
                
                if closing_speed > 0:
                    capped_closing_speed = np.clip(closing_speed, 0.0, 300.0)
                    
                    if dz < -50.0:
                        # 引入平滑衰减系数：从 dz = -50(系数1.0) 线性衰减到 dz = -250(系数0.0)
                        ramming_multiplier = np.clip((250.0 + dz) / 200.0, 0.0, 1.0)
                        reward_A_ramming += capped_closing_speed * 0.08 * dt * ramming_multiplier
                    else:
                        reward_A_ramming += capped_closing_speed * 0.08 * dt
                    
                    # 当进入末端抵近阶段 (400m 以内) 时，启动高精度的“动能撞击/导弹引导”逻辑
                    if new_dist <= TERMINAL_RADIUS: 
                        # 【优化】：彻底移除 z_alignment_factor
                        # 鼓励包含垂直俯冲在内的所有全向 3D 动能撞击
                        reward_A_ramming += closing_speed * 0.15 * dt
                # =========================================================================
                              
                # 单帧结算
                total_rewards["attacker_0"] += (
                    reward_A_progress 
                    + reward_A_tracking 
                    + reward_A_time 
                    + reward_A_ramming 
                    + reward_A_z_advantage     # 更新后的高度优势奖励
                    + reward_A_ground_warning 
                    + reward_A_energy_loss     # 新增的能量机动惩罚
                )

            # [角色 2] 目标机 (Evader) 奖励结算
            '''
            if "evader_0" in actions and not terminations["evader_0"]:
                # ================= Phase 1 打靶阶段简化 =================
                # 目标机作为固定靶，不再计算复杂的规避奖励，防止梯度混乱并节省算力
                total_rewards["evader_0"] += 0.0 
                # =======================================================
            '''
            if "evader_0" in actions and not terminations["evader_0"]:

                reward_E_survival = 0.5 * dt # 提高苟活底薪，鼓励多在天上待一秒是一秒
                reward_E_escape = 0.0
                reward_E_spoofing = 0.0
                
                if current_micro_dist <= self.warning_radius:
                    # 1. 逃逸奖励：成功拉开距离给予奖励
                    if micro_delta_dist > 0:
                        reward_E_escape = micro_delta_dist * 2.0 
                    
                    # 2. 角度破坏 (Spoofing)：破坏攻击机的瞄准
                    if cos_ata_attacker > 0.5: # 攻击机正看向我
                        # 惩罚目标机被锁定，逼迫它做急转弯脱离攻击机视线
                        reward_E_spoofing -= (cos_ata_attacker ** 2) * 5.0 * dt
                else:
                    pass

                # 目标机防地撞与防飞离边界硬性惩罚 (让它留在交战空域)
                reward_E_boundary = 0.0
                # 设定一个 1000m 的软地板警告线 (物理硬地板在 600m)
                if new_evader_pos[2] < 1000.0:
                    # 越靠近 600m，惩罚呈指数级上升
                    depth_ratio = (1000.0 - new_evader_pos[2]) / 400.0  # 比例 0.0 ~ 1.0
                    
                    # 【核心修改 1】：将基础惩罚从 20.0 暴增到 100.0，让它一旦跌破 1000m 就痛不欲生
                    reward_E_boundary -= (depth_ratio ** 2) * 100.0 * dt
                    
                    # 动态势能墙：如果在警告区内还往下掉，加重惩罚！
                    vz_E = trusted_states["evader_0"]["vel"][2]
                    if vz_E < -1.0:
                        reward_E_boundary -= abs(vz_E) * 3.0 * dt

                # 天花板同样提前设立软边界 (例如 3700m - 4000m)
                elif new_evader_pos[2] > 3700.0:
                    depth_ratio = (new_evader_pos[2] - 3700.0) / 300.0
                    reward_E_boundary -= (depth_ratio ** 2) * 20.0 * dt
                        
                total_rewards["evader_0"] += (reward_E_survival + reward_E_escape + reward_E_spoofing + reward_E_boundary)
            
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
            
            # 2. 真实的脱靶量 (CPA) 近炸结算
            # 【逻辑修正】：前置拦截时，CPA 瞬间目标极大概率已穿越 3-9 线进入后半球。
            # 因此，彻底移除 cos_ata_attacker > 0.0 的苛刻限制！只要进了 300 米圈且开始脱离，就是击杀。
            elif new_dist <= self.cpa_radius and raw_micro_delta > 0 and self.macro_step > 2:
                miss_distance = current_micro_dist

                # 保底给予 2000 分，距离越近额外奖励越高（最高再加 3000 分）
                linear_ratio = np.clip((self.cpa_radius - miss_distance) / (self.cpa_radius - 50.0), 0.0, 1.0)
                reward_terminal = 2000.0 + 3000.0 * linear_ratio
                
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
                    # 【修改】：目标机跌破 595 米即判定坠毁
                    death_floor = 10.0 if agent == "attacker_0" else 595.0
                    
                    if state[2] < death_floor:
                        crash_penalty = 2000.0 if agent == "attacker_0" else 5000.0
                        total_rewards[agent] -= crash_penalty
                        
                        terminations[agent] = True
                        infos[agent]["reason"] = "ground_crash"
                        crash_occurred = True
                    elif state[2] > 4900.0:
                        total_rewards[agent] -= 2000.0
                        terminations[agent] = True
                        infos[agent]["reason"] = "out_of_bounds" 
                        crash_occurred = True

            # 只要有飞机坠毁，立刻跳出微观物理循环
            if crash_occurred:
                break

        # --- 退出 Frame Skip 循环，结算当前决策步的最终结果 ---

        # 宏观战术趋势奖励结算
        if "attacker_0" in total_rewards and not terminations.get("attacker_0", False):
            # 1. 获取动作执行完毕后的最终状态
            final_A_state = self.pyb_env._getDroneStateVector(attacker_id)
            final_E_state = self.pyb_env._getDroneStateVector(evader_id)
            
            final_pos_A = final_A_state[0:3]
            final_pos_E = final_E_state[0:3]
            final_quat_A = final_A_state[3:7]
            
            # 2. 计算最终的视线向量和机头指向
            macro_los_dir = (final_pos_E - final_pos_A) / (np.linalg.norm(final_pos_E - final_pos_A) + 1e-6)
            rot_mat_final_A = p.getMatrixFromQuaternion(final_quat_A)
            final_forward_A = np.array([rot_mat_final_A[0], rot_mat_final_A[3], rot_mat_final_A[6]])
            
            # 3. 算出这个宏观步最终的 ATA 余弦值
            final_cos_ata = np.clip(np.dot(final_forward_A, macro_los_dir), -1.0, 1.0)
            
            # 4. 计算内的净变化量 (Delta)
            macro_delta_cos = final_cos_ata - getattr(self, 'last_cos_ata_A', final_cos_ata)
            
            # 5. 给予宏观趋势奖励并更新缓存
            if macro_delta_cos > 0:
                total_rewards["attacker_0"] += macro_delta_cos * 50.0 
                
            self.last_cos_ata_A = final_cos_ata
        
        # 判断是否超时 (Truncation)
        if (self.step_counter / self.CTRL_FREQ) > self.EPISODE_LEN_SEC:
            for agent in self.agents:
                truncations[agent] = True

            # 如果演习结束，且攻击机既没有坠毁也没有击杀（即苟活到了最后），给予巨额惩罚
            if not terminations.get("attacker_0", True) and "attacker_0" in total_rewards:
                total_rewards["attacker_0"] -= 2000.0  
                
            # 对应的，目标机成功拖延时间活到了最后，任务圆满完成，给予巨额奖励
            if not terminations.get("evader_0", True) and "evader_0" in total_rewards:
                total_rewards["evader_0"] += 500.0
        
        global_state_array = self._compute_global_state()
        observations = {} # 计算最新的观测值
        for agent in self.agents: # 注意：此时 self.agents 已经清理过了死掉的飞机
            observations[agent] = {
                "obs": self._compute_obs(agent),
                "global_state": global_state_array
            }
        
        # 对于刚刚在这一帧死亡的飞机，依然需要给它发送最后一次信息（包含死亡判定）
        for agent in self.possible_agents:
            if terminations[agent] or truncations[agent]:
                if agent not in observations:
                    observations[agent] = {
                        "obs": np.zeros(19, dtype=np.float32),
                        "global_state": global_state_array # 死亡瞬间依然让 Critic 看到全局
                    }

        # 必须清理掉本回合死亡的智能体
        self.agents = [
            a for a in self.agents
            if not (terminations[a] or truncations[a])
        ]

        # --- 计算全局结束标志 ---
        # 在 1v1 中，只要有任何一方死亡或超时，整局对抗立刻结束
        terminations["__all__"] = any(terminations.values()) if terminations else True
        truncations["__all__"] = any(truncations.values()) if truncations else True

        if terminations["__all__"] or truncations["__all__"]:
            if getattr(self, 'record_tacview', False):
                self.tacview_logger.close()

        # 记录 Tacview 帧 
        if getattr(self, 'record_tacview', False):
            current_time = self.step_counter / self.CTRL_FREQ
            self._record_tacview_frame(time_sec=current_time)

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

            # ================= 更新雷达标记点 =================
            if hasattr(self, 'radar_marker_A'):
                p.resetBasePositionAndOrientation(self.radar_marker_A, cur_attacker_pos, [0, 0, 0, 1], physicsClientId=self.pyb_env.CLIENT)
            if hasattr(self, 'radar_marker_E'):
                p.resetBasePositionAndOrientation(self.radar_marker_E, cur_evader_pos, [0, 0, 0, 1], physicsClientId=self.pyb_env.CLIENT)
            # =================================================
            
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

            # ================= HUD 文字与战术几何计算 =================
            # 1. 获取双方实时状态
            state_A = self.pyb_env._getDroneStateVector(attacker_id)
            state_E = self.pyb_env._getDroneStateVector(evader_id)
            
            pos_A, vel_A = state_A[0:3], state_A[10:13]
            pos_E, vel_E = state_E[0:3], state_E[10:13]
            
            alt_A, speed_A = pos_A[2], np.linalg.norm(vel_A)
            alt_E, speed_E = pos_E[2], np.linalg.norm(vel_E)

            # 2. 计算真实的战术夹角
            los_vec = pos_E - pos_A
            dist_cam = np.linalg.norm(los_vec)
            los_dir = los_vec / (dist_cam + 1e-6)
            
            # 提取主机真实机头指向 (通过四元数转旋转矩阵的第一列)
            rot_mat_A = p.getMatrixFromQuaternion(state_A[3:7])
            forward_A = np.array([rot_mat_A[0], rot_mat_A[3], rot_mat_A[6]])
            
            # 真实 ATA (天线偏角): 机头指向与视线的夹角
            ata_deg = np.degrees(np.arccos(np.clip(np.dot(forward_A, los_dir), -1.0, 1.0)))
            
            # 碰撞角偏差 (Collision Error): 相对速度与视线的夹角
            rel_vel = vel_A - vel_E
            rel_vel_dir = rel_vel / (np.linalg.norm(rel_vel) + 1e-6)
            collision_err_deg = np.degrees(np.arccos(np.clip(np.dot(rel_vel_dir, los_dir), -1.0, 1.0)))

            # 提取目标机滚转角 (观察 2G 盘旋是否保持在完美的 60 度)
            roll_E_deg = np.degrees(state_E[7])

            # 3. 极简单行字符串设计 (杜绝任何多行重叠隐患)
            # 主机：距离、ATA、碰撞偏差角
            hud_A_text = f"[A] Dist:{dist_cam:.0f}m | Spd:{speed_A:.0f}m/s | ATA:{ata_deg:.1f}* | Coll:{collision_err_deg:.1f}*"
            # 目标机：空速、滚转角 (用来监控盘旋靶是否正常飞)
            hud_E_text = f"[E] Spd:{speed_E:.0f}m/s | Roll:{roll_E_deg:.0f}*"

            # 4. 绑定与绘制
            drone_id_A = self.pyb_env.DRONE_IDS[0] if hasattr(self.pyb_env, 'DRONE_IDS') else self.pyb_env.drone_ids[0]
            drone_id_E = self.pyb_env.DRONE_IDS[1] if hasattr(self.pyb_env, 'DRONE_IDS') else self.pyb_env.drone_ids[1]

            # 初始化 ID 占位符
            if not hasattr(self, 'hud_A_id'): self.hud_A_id = -1
            if not hasattr(self, 'hud_E_id'): self.hud_E_id = -1

            # 将文字挂在飞机正上方 2.5 米处
            if self.hud_A_id == -1:
                self.hud_A_id = p.addUserDebugText(hud_A_text, [0, 0, 2.5], textColorRGB=[0.1, 0.4, 1.0], textSize=1.2, parentObjectUniqueId=drone_id_A, physicsClientId=self.pyb_env.CLIENT)
            else:
                self.hud_A_id = p.addUserDebugText(hud_A_text, [0, 0, 2.5], textColorRGB=[0.1, 0.4, 1.0], textSize=1.2, parentObjectUniqueId=drone_id_A, replaceItemUniqueId=self.hud_A_id, physicsClientId=self.pyb_env.CLIENT)
            
            if self.hud_E_id == -1:
                self.hud_E_id = p.addUserDebugText(hud_E_text, [0, 0, 2.5], textColorRGB=[1.0, 0.2, 0.2], textSize=1.2, parentObjectUniqueId=drone_id_E, physicsClientId=self.pyb_env.CLIENT)
            else:
                self.hud_E_id = p.addUserDebugText(hud_E_text, [0, 0, 2.5], textColorRGB=[1.0, 0.2, 0.2], textSize=1.2, parentObjectUniqueId=drone_id_E, replaceItemUniqueId=self.hud_E_id, physicsClientId=self.pyb_env.CLIENT)
            
            # 视线连线
            p.addUserDebugLine(cur_attacker_pos, cur_evader_pos, [0, 1, 1], 1.5, 1.5 / self.CTRL_FREQ, physicsClientId=self.pyb_env.CLIENT)
            
            # 绘制动态高度投影线 (直达海平面 Z=0)
            # 无论镜头拉多远，都能通过这根“柱子”看清飞机在全局的位置
            p.addUserDebugLine(cur_attacker_pos, [cur_attacker_pos[0], cur_attacker_pos[1], 0.0], [0.1, 0.4, 1.0], 1.5, 1.5 / self.CTRL_FREQ, physicsClientId=self.pyb_env.CLIENT)
            p.addUserDebugLine(cur_evader_pos, [cur_evader_pos[0], cur_evader_pos[1], 0.0], [1.0, 0.2, 0.2], 1.5, 1.5 / self.CTRL_FREQ, physicsClientId=self.pyb_env.CLIENT)

            # 计算两架飞机的空间中点 (Midpoint)
            mid_pos = (cur_attacker_pos + cur_evader_pos) / 2.0

            # 提取双方真实姿态用于镜头对齐
            attacker_rpy = self.pyb_env._getDroneStateVector(attacker_id)[7:10]
            evader_rpy = self.pyb_env._getDroneStateVector(evader_id)[7:10]
            smooth_yaw_A = np.degrees(attacker_rpy[2]) 
            smooth_yaw_E = np.degrees(evader_rpy[2]) 

            # 相机平滑跟随缓动
            self.cam_pos = self.cam_pos * 0.9 + cur_attacker_pos * 0.1

            if self.camera_mode == 1:
                # Mode 1: 经典第三人称尾随视角
                p.resetDebugVisualizerCamera(100.0, smooth_yaw_A - 90, -10, self.cam_pos, physicsClientId=self.pyb_env.CLIENT)
            
            elif self.camera_mode == 2:
                # Mode 2: 战术俯视地图 (Top-down Tactical Map)
                # 距离动态适应，但增加下限防止过近，上限限制在 8000 保证可视度
                tactical_dist = np.clip(dist_cam * 1.5, 4000.0, 8000.0) 
                p.resetDebugVisualizerCamera(tactical_dist, 0, -89.9, mid_pos, physicsClientId=self.pyb_env.CLIENT)
            
            elif self.camera_mode == 3:
                # Mode 3: 动态狗斗视角 (Over-the-shoulder)
                view_yaw = np.degrees(np.arctan2(cur_evader_pos[1] - cur_attacker_pos[1], cur_evader_pos[0] - cur_attacker_pos[0]))
                dynamic_dist = np.clip(dist_cam * 1.2, 150.0, 4000.0) 
                p.resetDebugVisualizerCamera(dynamic_dist, view_yaw - 90, -15, mid_pos, physicsClientId=self.pyb_env.CLIENT)
            
            elif self.camera_mode == 4:
                # Mode 4: 目标锁定抵近视角 (Target Tracking)
                # 修复：提取目标机的 Yaw 角进行追踪
                p.resetDebugVisualizerCamera(200.0, smooth_yaw_E - 90, -20, cur_evader_pos, physicsClientId=self.pyb_env.CLIENT)
            
            elif self.camera_mode == 5:
                # Mode 5: 全局大尺度远景 (God's Eye)
                # 将锚点从静态原点改为两机中点，确保交战空域永远在画面正中心
                # 借助新增的垂直投影线，即使飞机变成小点也能清晰辨别
                god_view_dist = np.clip(dist_cam * 2.5, 6000.0, 10000.0)
                p.resetDebugVisualizerCamera(god_view_dist, 45, -30, mid_pos, physicsClientId=self.pyb_env.CLIENT)

        return observations, total_rewards, terminations, truncations, infos
    
    def _record_tacview_frame(self, time_sec):
        """内部辅助函数：提取状态并传递给 Logger"""
        # _getDroneStateVector 返回的 19 维向量中：[0:3]是坐标XYZ, [7:10]是欧拉角RPY
        attacker_state_vec = self.pyb_env._getDroneStateVector(0)
        evader_state_vec = self.pyb_env._getDroneStateVector(1)
        
        state_A = {'pos': attacker_state_vec[0:3], 'rpy': attacker_state_vec[7:10]}
        state_E = {'pos': evader_state_vec[0:3], 'rpy': evader_state_vec[7:10]}
        
        self.tacview_logger.log_frame(time_sec, state_A, state_E)
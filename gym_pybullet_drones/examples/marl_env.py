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
        self.EPISODE_LEN_SEC = 60 # 回合最大时长
        self.cpa_radius = 400.0     # 近炸引信触发半径

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

        # 动作空间：3维连续变量 [-1.0, 1.0]
        # Action[0]: 切向加速度 (控制推力/减速板)
        # Action[1]: 法向过载 (控制俯仰拉杆)
        # Action[2]: 滚转角 (控制副翼)
        self.action_spaces = {
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
            for agent in self.possible_agents
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
            # Stage 1: 近距超视距 (新手村)
            self.d_min, self.d_max = 200.0, 600.0
            self.z_min, self.z_max = 1500.0, 2000.0
        elif self.curriculum_stage == 2:
            # Stage 2: 中距拉锯
            self.d_min, self.d_max = 400.0, 900.0
            self.z_min, self.z_max = 1800.0, 2500.0
        else:
            # Stage 3: 长程高空对决 (毕业期)
            self.d_min, self.d_max = 700.0, 1500.0
            self.z_min, self.z_max = 2200.0, 3200.0
    def _compute_global_state(self):
        """
        为 MAPPO 的 Critic 提取全知全能的全局状态 (Global State)
        """
        global_state = []
        
        # 遍历所有可能存在的飞机 (注意使用 self.possible_agents 保证顺序和维度固定)
        for i, agent in enumerate(self.possible_agents):
            if agent in self.agents:
                # 【核心修复点】
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
        evader_z = np.random.uniform(self.z_min - 500.0, self.z_min)
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
            if agent == "attacker_0":
                dx = -initial_pos[0]
                dy = -initial_pos[1]
                yaw = np.arctan2(dy, dx)
                self.attacker_init_yaw = yaw # 记录一下攻击机的朝向
            else:
                # 强制战术夹角为 0 (纯尾追)
                tactical_offset = 0.0 
                # 保留 ±10度的微小扰动，防止过拟合
                noise = np.random.uniform(-np.pi/18, np.pi/18)
                yaw = self.attacker_init_yaw + tactical_offset + noise

                '''
                # ================= 课程学习 Stage 1.5：全向直线拦截 =================
                # 引入四种经典的战术初始态势，并加入 ±15度 的随机扰动防止过拟合
                
                # 0:        纯尾追 (Tail-on)
                # np.pi/2:  左侧向交叉 (Left-Beam)
                # -np.pi/2: 右侧向交叉 (Right-Beam)
                # np.pi:    迎头对冲 (Head-on)
                tactical_offset = np.random.choice([0.0, np.pi/2, -np.pi/2, np.pi])
                
                # 添加随机扰动 (约 ±15 度)
                noise = np.random.uniform(-np.pi/12, np.pi/12)
                yaw = self.attacker_init_yaw + tactical_offset + noise
                # ====================================================================
                '''

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
        # 初始动作占位符必须是 3 维 NumPy 零向量
        self.last_actions = {agent: np.zeros(3, dtype=np.float32) for agent in self.agents}
    
        # 计算开局时的初始距离 (用于第一帧的奖励计算基准)
        attacker_pos = self.pyb_env._getDroneStateVector(0)[0:3]
        evader_pos = self.pyb_env._getDroneStateVector(1)[0:3]
        self.prev_dist = np.linalg.norm(attacker_pos - evader_pos)

        # 记录上一帧的 ATA 余弦值，用于计算趋势
        self.last_cos_ata_A = 1.0

        # ================= 课程学习 Stage 1.0：移动打靶 =================
        # 【降维】强制目标机只能直飞
        self.evader_maneuver = "straight"

        '''
        # ================= 课程学习 Stage 2：随机化目标机盘旋 =================
        # 随机决定本回合目标机的机动策略。
        # 概率分布：40% 直飞，30% 左转，30% 右转
        self.evader_maneuver = np.random.choice(
            ["straight", "turn_left", "turn_right"], 
            p=[0.4, 0.3, 0.3]
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
        MAX_DIST = 5000.0     
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

        # --- [修改后] ---
        # 动作平滑度惩罚 (连续动作空间专属)
        for agent, act in actions.items():
            last_act = self.last_actions.get(agent, np.zeros(3, dtype=np.float32))
            
            # 计算这一帧和上一帧推杆动作的差异大小 (欧氏距离 L2 Norm)
            action_delta = np.linalg.norm(act - last_act)
            
            # 根据猛推摇杆的剧烈程度给予惩罚 (系数 0.1 比较温和，鼓励丝滑微调)
            total_rewards[agent] -= 0.1 * action_delta 
            
            # 存入本帧动作，必须使用 .copy() 防止内存地址的引用污染
            self.last_actions[agent] = np.array(act).copy()

        attacker_state_init = self.pyb_env._getDroneStateVector(attacker_id)
        evader_state_init = self.pyb_env._getDroneStateVector(evader_id)

        dist = np.linalg.norm(attacker_state_init[0:3] - evader_state_init[0:3])
        current_micro_dist = dist

        self.macro_step += 1 # 新增：每次 AI 下达指令，宏观步数推进 1 步

        # 绝对信任的本地物理账本
        # 彻底抛弃每帧从 PyBullet 读取速度的逻辑，防止无人机的空气阻力污染数据！
        trusted_states = {
            "attacker_0": {"pos": attacker_state_init[0:3].copy(), "vel": attacker_state_init[10:13].copy()},
            "evader_0":   {"pos": evader_state_init[0:3].copy(),   "vel": evader_state_init[10:13].copy()}
        }

        for _ in range(dynamic_frame_skip):           
            self.pyb_env.step(np.zeros((2, 4)))
            self.step_counter += 1
            
            for i, agent in enumerate(["attacker_0", "evader_0"]):
                if agent not in actions or terminations[agent]: # 如果这架飞机已经判定死亡，则直接跳过
                    continue

                pyb_id = self.pyb_env.DRONE_IDS[i] if hasattr(self.pyb_env, 'DRONE_IDS') else self.pyb_env.drone_ids[i]
                
                # ================= 新增：连续动作解包与线性映射 =================
                # 确保获取的是 3 维 NumPy 数组，并且限制在 [-1, 1] 之间以防异常值
                action_vec = np.clip(actions[agent], -1.0, 1.0)
                
                # 1. 切向过载 (n_x): 映射到 [-2.0, 2.0]
                n_x_cmd = action_vec[0] * 2.0
                
                # 2. 法向过载 (n_n): 映射到 [MIN_G, MAX_G]
                # 公式: MIN + (MAX - MIN) * (val + 1) / 2
                n_n_cmd = self.MIN_G + (self.MAX_G - self.MIN_G) * (action_vec[1] + 1.0) / 2.0
                
                # 3. 滚转角 (mu): 映射到 [-180度, 180度] 即 [-pi, pi]
                mu_cmd = action_vec[2] * np.pi
                # ================================================================

                # ================= Phase 2 干预：注入完美的水平盘旋 =================
                if agent == "evader_0":
                    if self.evader_maneuver == "turn_left":
                        n_x_cmd = 0.0          # 保持匀速
                        n_n_cmd = 2.0          # 2G 法向过载
                        mu_cmd = np.pi / 3.0   # 60度滚转 (保持高度不掉)
                    elif self.evader_maneuver == "turn_right":
                        n_x_cmd = 0.0
                        n_n_cmd = 2.0
                        mu_cmd = -np.pi / 3.0  # 向右 60度滚转
                    else: # 直飞
                        n_x_cmd = 0.0
                        n_n_cmd = 1.0
                        mu_cmd = 0.0
                # ====================================================================
                
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
                # 如果低于 800 米，且具有超过 5m/s 的下坠速度，强制接管
                if agent_current_z < 800.0 and vel[2] < -5.0:  
                    n_n_cmd = current_max_g  # 强制给足最大过载拉起
                    mu_cmd = 0.0             # 强制改平

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
                new_V = np.clip(new_V, self.STALL_SPEED, current_max_speed)
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
                    if new_pos[2] > 5000.0 :
                        new_pos[2] = 5000.0
                        total_rewards[agent] -= 0.5 * dt  # 累加高度软惩罚
                    # 正常防钻地（不给惩罚，直接限制）
                    elif new_pos[2] < 1.0:
                        new_pos[2] = 1.0
                else:
                    # 目标机：仅保留原有的物理边界限制，不施加任何额外惩罚
                    new_pos[2] = np.clip(new_pos[2], 1.0, 5000.0)

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

            # 更新 gym-pybullet-drones 的内置缓存...
            if hasattr(self.pyb_env, '_updateAndStoreKinematicInformation'):
                self.pyb_env._updateAndStoreKinematicInformation()

            # 重新提取一次绝对干净的物理状态...
            new_attacker_state = self.pyb_env._getDroneStateVector(attacker_id)

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
                TERMINAL_RADIUS = 800.0  # 定义末端冲刺阶段的判定半径

                # 计算双方的高度差 (Z轴距离)
                dz = new_attacker_pos[2] - new_evader_pos[2]

                # 1. 靠近奖励 (全局生效：缩短距离加分，被拉开扣分)
                reward_A_progress = -micro_delta_dist * 2.0 
                reward_A_progress = np.clip(reward_A_progress, -10.0, 10.0)

                # 2. 时间惩罚 (全局生效：逼迫速战速决)
                reward_A_time = -1.0 * dt

                reward_A_z_advantage = 0.0
                if dz > 0:
                    # 主机在上方：给予持续的正向能量奖励 (势能储备)
                    reward_A_z_advantage = 0
                else:
                    # 主机在下方：给予较重的惩罚，逼迫它拉起机头爬升
                    reward_A_z_advantage = dz * 0.005 * dt
                
                reward_A_energy_loss = 0.0 

                # reward_A_energy_loss = -((n_n - 1.0) ** 2) * 0.08 * dt  # 新增能量管理惩罚

                # 攻击机软地板警告 
                reward_A_ground_warning = 0.0
                if new_attacker_pos[2] < 1000.0:  
                    # 高度越低，惩罚呈指数级上升
                    depth_ratio = (1000.0 - new_attacker_pos[2]) / 1000.0
                    reward_A_ground_warning = -(depth_ratio ** 2) * 5.0 * dt

                    # 提取当前 Z 轴速度 (垂直速度)
                    vz = trusted_states["attacker_0"]["vel"][2]
                    
                    # 【核心保命机制】如果处于低空，且还在向下掉高度
                    if vz < -1.0: 
                        # 下坠越快，乘法叠加的惩罚越极端 (动态势能墙)
                        # 例如 vz = -50m/s 时，每秒扣除巨大的分数，逼迫网络产生对“死亡俯冲”的恐惧
                        reward_A_ground_warning -= abs(vz) * 0.2 * dt
                
                reward_A_tracking = 0.0
                reward_A_ramming = 0.0

                if new_dist <= TERMINAL_RADIUS:
                    # --- 末端冲刺阶段 (Terminal Phase) ---
                    # 1. 取消 ATA 瞄准惩罚，彻底释放机动限制
                    reward_A_tracking = 0.0 
                    
                    # 2. 动能冲刺奖励 (Ramming Bonus)
                    # 替换绝对速度奖励为接近速度奖励
                    los_vec = new_evader_pos - new_attacker_pos
                    los_dir = los_vec / (np.linalg.norm(los_vec) + 1e-6)
                    attacker_vel = self.pyb_env._getDroneStateVector(attacker_id)[10:13]
                    
                    # 引入水平冲刺系数 
                    # 避免主机在最后一刻从天顶垂直“砸”向目标。只有当高度差极小时，才给予 100% 的速度冲刺奖励。高度差越大，冲刺奖励的折扣越狠。
                    z_alignment_factor = np.clip((200.0 - abs(dz)) / 200.0, 0.0, 1.0)
                    # 计算速度在视线方向上的投影 (接近率)
                    closing_speed = np.dot(attacker_vel, los_dir)
                    if closing_speed > 0:
                        reward_A_ramming = closing_speed * 0.05 * dt * z_alignment_factor
                    else:
                        reward_A_ramming = 0.0
                    # ========================================================
                else:
                    # --- 中程追踪阶段 (Mid-course Phase) ---
                    
                    # 计算相对速度矢量，用于指引“直线提前量拦截”
                    vel_A = trusted_states["attacker_0"]["vel"]
                    vel_E = trusted_states["evader_0"]["vel"]
                    rel_vel = vel_A - vel_E
                    rel_vel_dir = rel_vel / (np.linalg.norm(rel_vel) + 1e-6)
                    
                    # cos_collision 衡量的是“相对速度”是否指向目标，这是直线拦截的核心！
                    cos_collision = np.clip(np.dot(rel_vel_dir, los_dir), -1.0, 1.0)

                    # ==========================================================
                    # BFM 综合战术几何奖励 (ATA + AA + HCA + Collision)
                    # ==========================================================
                    reward_A_tracking = 0.0
                    
                    # 1. 基础瞄准：机头必须试图看向敌机 (ATA)
                    if cos_ata_attacker > 0.866:   # 30度内
                        reward_A_tracking += 3.0 * dt
                    elif cos_ata_attacker > 0.0:   # 前半球
                        reward_A_tracking += cos_ata_attacker * 1.5 * dt
                    else:
                        reward_A_tracking -= 3.0 * dt # 严厉惩罚背对目标
                        
                    # 2. 阵位优势：必须试图进入敌机后半球 (AA)
                    # cos_aa_attacker 越大，说明越靠近敌机正后方的 6 点钟盲区
                    if cos_aa_attacker > 0.5: # 处于敌机后半球 60 度扇区
                        reward_A_tracking += cos_aa_attacker * 2.0 * dt
                    elif cos_aa_attacker < -0.5: # 处于敌机正前方危险区
                        reward_A_tracking -= 1.0 * dt

                    # 3. 速度矢量对齐：防止交臂过冲 (HCA)
                    # 同向飞行能极大降低相对闭合率，提供更充裕的击杀窗口
                    if cos_hca > 0.866: # 航向差异小于 30 度
                        reward_A_tracking += 2.0 * dt

                    # 4. 碰撞截击：引导直线提前量 (Collision)
                    if cos_collision > 0.95: 
                        reward_A_tracking += 5.0 * dt
                        
                    # 5. 终极协同分：进入“黄金控制区” (Control Zone)
                    # 必须同时满足：机头对准 (ATA)、在敌机屁股后面 (AA)、且同向飞行 (HCA)
                    if cos_ata_attacker > 0.866 and cos_aa_attacker > 0.866 and cos_hca > 0.866:
                        reward_A_tracking += 15.0 * dt # 给予极高的reward
               
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
            if "evader_0" in actions and not terminations["evader_0"]:
                # ================= Phase 1 打靶阶段简化 =================
                # 目标机作为固定靶，不再计算复杂的规避奖励，防止梯度混乱并节省算力
                total_rewards["evader_0"] += 0.0 
                # =======================================================
            '''
            if "evader_0" in actions and not terminations["evader_0"]:
                WARNING_RADIUS = 500.0  # 告警半径设置
                
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
                    # 1. 动作级奖励：鼓励选择平飞 BFM
                    # 0=匀速直飞, 1=加速直飞, 2=减速直飞
                    if evader_action in [0, 1, 2]: 
                        reward_E_straight += 1.0 * dt  # 提高奖励权重，明确告诉AI这是对的
                    else:
                        # 惩罚在安全距离乱做大过载或滚转机动
                        reward_E_straight -= 0.5 * dt

                    # 2. 物理姿态级惩罚：逼迫飞机保持水平
                    # 提取目标机当前的 Roll (滚转) 和 Pitch (俯仰) 角
                    evader_rpy = new_evader_state[7:10] 
                    roll = evader_rpy[0]
                    pitch = evader_rpy[1]
                    
                    # 姿态越倾斜，扣分越多 (鼓励 Roll 和 Pitch 趋近于 0)
                    attitude_penalty = (abs(roll) + abs(pitch)) * 0.5 * dt
                    reward_E_straight -= attitude_penalty

                    # 3. 垂直速度惩罚：替代原本僵硬的“绝对高度惩罚”
                    # 只要飞机不往下掉，就不扣分。这能有效防止死亡俯冲。
                    evader_vel_z = new_evader_state[10:13][2]
                    if evader_vel_z < -2.0:  # 允许 2m/s 以内的微小掉高，超过则惩罚下坠率
                        reward_E_straight -= abs(evader_vel_z) * 0.05 * dt

                # 目标机软地板警告 
                reward_E_ground_warning = 0.0
                if new_evader_pos[2] < 300.0:
                    reward_E_ground_warning = -(300.0 - new_evader_pos[2]) * 0.5 * dt
                        
                # 单帧结算
                total_rewards["evader_0"] += (reward_E_escape + reward_E_survival + reward_E_jinking + reward_E_straight + reward_E_ground_warning)

                '''
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
            # 如果进入杀伤圈，且距离开始拉大，说明刚刚掠过极小值点
            elif new_dist < self.cpa_radius and raw_micro_delta > 0:
                miss_distance = new_dist - raw_micro_delta # 取上一微小帧的极小值
                
                # 根据脱靶量计算梯度得分：基础分1000 + 4000 * (1 - (脱靶量 - 50.0) / 杀伤区间)
                score_ratio = 1.0 - ((miss_distance - 50.0) / (self.cpa_radius - 50.0))
                reward_terminal = 1000.0 + 4000.0 * np.clip(score_ratio, 0.0, 1.0)
                
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
                        total_rewards[agent] -= 5000.0
                        terminations[agent] = True
                        infos[agent]["reason"] = "ground_crash"
                        crash_occurred = True 
                    elif state[2] > 5000.0:
                        total_rewards[agent] -= 5000.0
                        terminations[agent] = True
                        infos[agent]["reason"] = "out_of_bounds" 
                        crash_occurred = True

            # 只要有飞机坠毁，立刻跳出微观物理循环
            if crash_occurred:
                break

        # --- 退出 Frame Skip 循环，结算当前决策步的最终结果 ---

        # 【新增：宏观战术趋势奖励结算】
        if "attacker_0" in total_rewards and not terminations.get("attacker_0", False):
            # 1. 获取 0.2 秒动作执行完毕后的最终状态
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
            
            # 4. 计算 0.2 秒内的净变化量 (Delta)
            macro_delta_cos = final_cos_ata - getattr(self, 'last_cos_ata_A', final_cos_ata)
            
            # 5. 给予宏观趋势奖励并更新缓存
            if macro_delta_cos > 0:
                # 既然是 0.2 秒的积累量，这里的权重可以适当给大一点 (比如 20.0)
                total_rewards["attacker_0"] += macro_delta_cos * 20.0 
                
            self.last_cos_ata_A = final_cos_ata
        
        # 判断是否超时 (Truncation)
        if (self.step_counter / self.CTRL_FREQ) > self.EPISODE_LEN_SEC:
            for agent in self.agents:
                truncations[agent] = True

            # 如果演习结束，且攻击机既没有坠毁也没有击杀（即苟活到了最后），给予巨额惩罚
            if not terminations.get("attacker_0", True) and "attacker_0" in total_rewards:
                total_rewards["attacker_0"] -= 500.0  # 减轻超时惩罚，鼓励先生存再输出
                
            # 对应的，目标机成功拖延时间活到了最后，任务圆满完成，给予巨额奖励
            if not terminations.get("evader_0", True) and "evader_0" in total_rewards:
                total_rewards["evader_0"] += 3000.0
        
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
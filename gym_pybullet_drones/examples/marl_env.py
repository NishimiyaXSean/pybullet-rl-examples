import numpy as np
import gymnasium as gym
import pybullet as p
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

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
        self.EPISODE_LEN_SEC = 100 # 回合最大时长
        self.SMOOTH_FACTOR = 0.1  # PID 轨迹平滑系数
        self.cpa_radius = 1.0     # 近炸引信触发半径

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

        # 2. 在该象限内，随机生成 3.0 到 8.0 米的安全边界距离
        attacker_x = sign_x * np.random.uniform(3.0, 8.0)
        attacker_y = sign_y * np.random.uniform(3.0, 8.0)
        attacker_z = np.random.uniform(1.5, 4.0) # 高度也适当随机

        # 3. 目标机强制取相反符号，确保永远出生在对角象限！
        evader_x = -sign_x * np.random.uniform(3.0, 8.0)
        evader_y = -sign_y * np.random.uniform(3.0, 8.0)
        evader_z = np.random.uniform(2.0, 5.0)

        # 组合成新的初始坐标数组
        new_init_xyzs = np.array([
            [attacker_x, attacker_y, attacker_z],
            [evader_x, evader_y, evader_z]
        ])

        # 覆盖 gym-pybullet-drones 底层环境缓存的初始坐标
        self.pyb_env.INIT_XYZS = new_init_xyzs
        
        # 重置底层物理引擎
        raw_obs, _ = self.pyb_env.reset()

        # 初始化时间步与两架飞机的局部追踪变量
        self.step_counter = 0
        self.target_yaws = {"attacker_0": 0.0, "evader_0": 0.0}
        self.user_input_pos = {}
        self.current_target_pos = {}
        
        # 为每架飞机提取真实的初始物理位置
        for i, agent in enumerate(self.agents):
            initial_pos = self.pyb_env._getDroneStateVector(i)[0:3]
            self.user_input_pos[agent] = initial_pos.copy()
            self.current_target_pos[agent] = initial_pos.copy()

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
            
            # 为目标机生成一个半透明的近炸引信杀伤圈
            fuze_v_id = p.createVisualShape(p.GEOM_SPHERE, radius=self.cpa_radius, rgbaColor=[1, 0.5, 0, 0.25])
            self.fuze_obj_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=fuze_v_id, basePosition=evader_pos, physicsClientId=self.pyb_env.CLIENT)

            z_offset = 0.05
            # 绘制主坐标轴
            p.addUserDebugLine([-15, 0, z_offset],[15, 0, z_offset], [1, 0, 0], 4, 0, physicsClientId=self.pyb_env.CLIENT)
            p.addUserDebugLine([0, -15, z_offset], [0, 15, z_offset],[0, 1, 0], 4, 0, physicsClientId=self.pyb_env.CLIENT)
            p.addUserDebugLine([0, 0, z_offset],[0, 0, 15], [0, 0.5, 1], 4, 0, physicsClientId=self.pyb_env.CLIENT)
            
            '''
            # 绘制底层地平面雷达网格 (Z=0)，范围 -15 到 15
            for i in range(-14, 15, 2):
                if i != 0: 
                    grid_color = [0.1, 0.2, 0.3]      
                    p.addUserDebugLine([i, -15, 0.0], [i, 15, 0.0], grid_color, 1.0, 0, physicsClientId=self.pyb_env.CLIENT)
                    p.addUserDebugLine([-15, i, 0.0], [15, i, 0.0], grid_color, 1.0, 0, physicsClientId=self.pyb_env.CLIENT)
            '''
            
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
        my_vel = my_state[10:13]
        my_rpy = my_state[7:10]         # 自身姿态 (Roll, Pitch, Yaw)
        my_ang_vel = my_state[13:16]    # 自身角速度
        my_z_height = my_state[2]       # 自身绝对高度

        enemy_pos = enemy_state[0:3]
        enemy_vel = enemy_state[10:13]

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

        # --- E. 我当前的虚拟引导点 ---
        world_virt_pos = self.current_target_pos[agent] - my_pos
        local_virt_pos, _ = p.multiplyTransforms([0, 0, 0], inv_quat, world_virt_pos, [0, 0, 0, 1])
        local_virt_pos = np.array(local_virt_pos)

        # 5. 物理量级缩放 (Pre-normalization) - 防止神经网络梯度爆炸
        MAX_DIST = 15.0     
        MAX_VEL = 5.0       
        MAX_ANG_VEL = 2 * np.pi 

        norm_local_rel_pos = local_rel_pos / MAX_DIST
        norm_local_vel = local_vel / MAX_VEL
        norm_rpy = my_rpy / np.pi                  
        norm_local_ang_vel = local_ang_vel / MAX_ANG_VEL
        norm_z_height = my_z_height / MAX_DIST     
        norm_local_virt_pos = local_virt_pos / MAX_DIST
        norm_local_enemy_vel = local_enemy_vel / MAX_VEL
        
        # 6. 拼接 19 维特征数组，形状严丝合缝
        obs_array = np.concatenate([
            norm_local_rel_pos,    # 3维: 敌机相对位置
            norm_local_vel,        # 3维: 我的空速
            norm_rpy,              # 3维: 我的姿态
            norm_local_ang_vel,    # 3维: 我的角速度
            [norm_z_height],       # 1维: 我的高度
            norm_local_virt_pos,   # 3维: PID指引点
            norm_local_enemy_vel   # 3维: 敌机速度矢量
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
                speed_multiplier = 0.6 if agent == "evader_0" else 1.0
                vx = (1.5 + n_x * 0.8) * speed_multiplier
                vy = 0.0
                vz = (n_n * np.cos(mu) - 1.0) * 0.15
                
                yaw_rate = abs(n_n) * np.sin(mu) * 0.5
                target_vel_local = np.array([vx, vy, vz])
                
                # 累加 Yaw 角度
                self.target_yaws[agent] += yaw_rate * dt

                # 生成一个没有俯仰(Pitch)和滚转(Roll)的纯净偏航姿态
                pilot_quat = p.getQuaternionFromEuler([0, 0, self.target_yaws[agent]])
                
                vel_world, _ = p.multiplyTransforms([0,0,0], pilot_quat, target_vel_local, [0,0,0,1])
                vel_world = np.array(vel_world)

                self.user_input_pos[agent] += vel_world * dt
                
                # ================= 修复：高度限制与天花板惩罚 =================
                if agent == "attacker_0":
                    # 我方无人机：触碰 10 米天花板时给予持续惩罚，防止利用边界“滑行”
                    if self.user_input_pos[agent][2] > 10.0:
                        self.user_input_pos[agent][2] = 10.0
                        total_rewards[agent] -= 0.5 * dt  # 累加高度软惩罚
                    # 正常防钻地（不给惩罚，直接限制）
                    elif self.user_input_pos[agent][2] < 1.0:
                        self.user_input_pos[agent][2] = 1.0
                else:
                    # 目标机：仅保留原有的物理边界限制，不施加任何额外惩罚
                    self.user_input_pos[agent][2] = np.clip(self.user_input_pos[agent][2], 1.0, 10.0)
                # ==========================================================

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

                # 相机跟随逻辑
                self.cam_pos = self.cam_pos * 0.95 + cur_attacker_pos * 0.05
                smooth_yaw = np.degrees(self.target_yaws["attacker_0"])

                if self.camera_mode == 1:
                    p.resetDebugVisualizerCamera(2.0, smooth_yaw - 90, -20, self.cam_pos, physicsClientId=self.pyb_env.CLIENT)
                elif self.camera_mode == 2:
                    p.resetDebugVisualizerCamera(12.0, 0, -89.9, [0, 0, 1.0], physicsClientId=self.pyb_env.CLIENT)
                elif self.camera_mode == 3:
                    p.resetDebugVisualizerCamera(max(2.0, dist_cam * 0.8), drone_angle - 45, -20, cur_evader_pos, physicsClientId=self.pyb_env.CLIENT)
                elif self.camera_mode == 4:
                    p.resetDebugVisualizerCamera(6.0, 45, -30, cur_evader_pos, physicsClientId=self.pyb_env.CLIENT)
                elif self.camera_mode == 5:
                    p.resetDebugVisualizerCamera(15.0, 90, -5, [0, 0, 3.0], physicsClientId=self.pyb_env.CLIENT)

            # 计算两架飞机的当前距离和相对位置
            rel_pos_world = evader_pos - attacker_pos
            dist = np.linalg.norm(rel_pos_world)

            # 计算距离变化率
            frame_delta_dist = dist - self.prev_dist
            self.prev_dist = dist

            # 计算攻击机的 ATA (天线前置角 - 是否对准目标)
            attacker_quat = p.getQuaternionFromEuler([0, 0, self.target_yaws["attacker_0"]])
            rot_mat_A = p.getMatrixFromQuaternion(attacker_quat)

            # 仅使用 XY 平面计算 ATA 夹角
            forward_vec_xy = np.array([rot_mat_A[0], rot_mat_A[3]])
            target_dir_xy = rel_pos_world[0:2] / (np.linalg.norm(rel_pos_world[0:2]) + 1e-6)
            cos_theta_xy = np.clip(np.dot(forward_vec_xy, target_dir_xy), -1.0, 1.0)
            ata_angle_attacker = np.arccos(cos_theta_xy)

            # [角色 1] 攻击机 (Attacker) 奖励结算
            if "attacker_0" in actions:
                TERMINAL_RADIUS = 3.0  # 定义末端冲刺阶段的判定半径 (可调参，建议 3.0~4.0 米)

                # 计算双方的高度差 (Z轴距离)
                dz = attacker_pos[2] - evader_pos[2]

                # 1. 靠近奖励 (全局生效：缩短距离加分，被拉开扣分)
                reward_A_progress = -frame_delta_dist * 20.0 
                
                # 2. 时间惩罚 (全局生效：逼迫速战速决)
                reward_A_time = -0.1 * dt

                # ================= 新增：Z 轴共面对齐惩罚 =================
                # 逼迫主机下降到与目标机大致平齐的高度。
                # 如果高度差大于 1.0 米，则开始施加基于高度差的持续惩罚
                reward_A_z_penalty = 0.0
                if abs(dz) > 1.0:
                    reward_A_z_penalty = -(abs(dz) - 1.0) * 1.5 * dt
                # ========================================================
                
                reward_A_tracking = 0.0
                reward_A_ramming = 0.0

                if dist <= TERMINAL_RADIUS:
                    # --- 末端冲刺阶段 (Terminal Phase) ---
                    # 1. 取消 ATA 瞄准惩罚，彻底释放机动限制
                    reward_A_tracking = 0.0 
                    
                    # 2. 动能冲刺奖励 (Ramming Bonus)
                    # 提取主机当前的绝对线速度，速度越快得分越高，鼓励“踩油门”撞击
                    attacker_vel = self.pyb_env._getDroneStateVector(attacker_id)[10:13]
                    vel_norm = np.linalg.norm(attacker_vel)
                    
                    # ================= 修改：引入水平冲刺系数 =================
                    # 避免主机在最后一刻从天顶垂直“砸”向目标。
                    # 只有当高度差极小 (比如小于 1.5 米) 时，才给予 100% 的速度冲刺奖励。
                    # 高度差越大，冲刺奖励的折扣越狠。
                    z_alignment_factor = np.clip(1.5 - abs(dz), 0.0, 1.0)
                    reward_A_ramming = vel_norm * 5.0 * dt * z_alignment_factor
                    # ========================================================
                else:
                    # --- 中程追踪阶段 (Mid-course Phase) ---
                    # 距离较远时，严抓航向对准
                    reward_A_tracking = -(ata_angle_attacker / np.pi) * 2.0 * dt
                
                # 单帧结算
                total_rewards["attacker_0"] += (reward_A_progress + reward_A_tracking + reward_A_time + reward_A_ramming)

            # [角色 2] 目标机 (Evader) 奖励结算
            if "evader_0" in actions:
                WARNING_RADIUS = 6.0  # 告警半径设置为 6 米
                
                reward_E_escape = 0.0
                reward_E_jinking = 0.0
                reward_E_straight = 0.0
                
                # 苟活奖励 (始终存在)
                reward_E_survival = 0.1 * dt
                
                rel_pos_xy = evader_pos[0:2] - attacker_pos[0:2]
                dist_xy = np.linalg.norm(rel_pos_xy)

                if dist_xy <= WARNING_RADIUS: # 使用水平距离判断是否触发告警
                    # --- 危险区域：激活逃逸与规避 ---
                    # 1. 逃逸奖励
                    reward_E_escape = frame_delta_dist * 15.0  
                    
                    # 2. 智能规避奖励 (Jinking)
                    if ata_angle_attacker < (np.pi / 6.0):
                        evader_action = int(actions["evader_0"])
                        _, n_n, mu = self.bfm_action_mapping[evader_action]
                        if abs(n_n) > 1.5 or abs(mu) > 0:
                            reward_E_jinking = 1.0 * dt
                else:
                    # --- 安全区域：鼓励直线平飞 ---
                    evader_action = int(actions["evader_0"])
                    # BFM 动作库中：0=匀速直飞, 1=加速直飞, 2=减速直飞
                    if evader_action in [0, 1, 2]: 
                        reward_E_straight = 0.5 * dt
                    else:
                        # 如果在安全距离做大过载机动，给予轻微惩罚
                        reward_E_straight = -0.3 * dt
                        
                # 单帧结算
                total_rewards["evader_0"] += (reward_E_escape + reward_E_survival + reward_E_jinking + reward_E_straight)

            # --- 碰撞与终止条件检测 (在每一小帧都要检测) ---
            new_attacker_state = self.pyb_env._getDroneStateVector(attacker_id)
            new_evader_state = self.pyb_env._getDroneStateVector(evader_id)
            new_dist = np.linalg.norm(new_attacker_state[0:3] - new_evader_state[0:3])

            # 1. 动能撞击 / 击杀成功
            if new_dist < 0.15:
                if "attacker_0" in total_rewards: total_rewards["attacker_0"] += 300.0
                if "evader_0" in total_rewards: total_rewards["evader_0"] -= 300.0
                terminations["attacker_0"] = True
                terminations["evader_0"] = True

                # 记录终端坐标 (放入 info 字典，供未来测试脚本绘图使用)
                infos["attacker_0"]["terminal_drone_pos"] = new_attacker_state[0:3].copy()
                infos["attacker_0"]["terminal_target_pos"] = new_evader_state[0:3].copy()
                infos["attacker_0"]["is_success"] = True
                break # 直接结束本轮 AI 决策的 repeat 循环

            # 2. 擦肩而过，触发近炸引信
            # new_dist 是物理步进后的距离，dist 是步进前的距离。
            # 如果进入杀伤圈，且距离开始拉大 (new_dist - dist > 0)，说明刚刚掠过极小值点
            elif new_dist < self.cpa_radius and (new_dist - dist) > 0:
                # 此时步进前的距离 dist 就是本次交锋的最小脱靶量
                miss_distance = dist 
                
                # 根据脱靶量计算梯度得分：基础分50 + 250 * (1 - (脱靶量 - 0.15) / 杀伤区间)
                score_ratio = 1.0 - ((miss_distance - 0.15) / (self.cpa_radius - 0.15))
                reward_terminal = 50.0 + 250.0 * score_ratio
                
                # 双方进行分数结算 (零和博弈)
                if "attacker_0" in total_rewards: total_rewards["attacker_0"] += reward_terminal
                if "evader_0" in total_rewards: total_rewards["evader_0"] -= reward_terminal
                
                terminations["attacker_0"] = True
                terminations["evader_0"] = True
                
                # 记录终端坐标
                infos["attacker_0"]["terminal_drone_pos"] = new_attacker_state[0:3].copy()
                infos["attacker_0"]["terminal_target_pos"] = new_evader_state[0:3].copy()
                infos["attacker_0"]["is_success"] = True
                break

            # 3. 地板/天空边界惩罚
            for agent, state in zip(["attacker_0", "evader_0"], [new_attacker_state, new_evader_state]):
                if agent in total_rewards: # 只有这个 agent 还在计分板上，才对它进行边界惩罚！
                    if state[2] < 0.1 or state[2] > 12.0:
                        total_rewards[agent] -= 100.0
                        terminations[agent] = True

        # --- 退出 Frame Skip 循环，结算当前决策步的最终结果 ---
        
        # 判断是否超时 (Truncation)
        if (self.step_counter / self.CTRL_FREQ) > self.EPISODE_LEN_SEC:
            for agent in self.agents:
                truncations[agent] = True
        
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
import os
import time
import numpy as np
import pybullet as p
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DronePIDEnv(CtrlAviary):
    def __init__(self, gui=False):
        self.target_pos = np.array([1.0, 1.0, 1.0])
        self.target_obj_id = -1

        # === 新增：动态目标属性 ===
        self.target_anchor = np.zeros(3) # 记录红球出生点 (用于限制活动范围)
        self.target_v = np.zeros(3)      # 红球的三维移动速度
        # === 新增：机动模式与圆周运动参数 ===
        self.target_mode = 0        # 0:X轴, 1:Y轴, 2:Z轴, 3:圆周
        self.target_angle = 0.0     # 圆周运动的当前极角
        self.target_omega = 0.0     # 圆周运动的角速度 (rad/s)
        self.target_radius = 1.5    # 圆周运动的半径

        self.prev_dist = 0.0
        self.camera_mode = 4  # 默认相机视角

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=np.array([[-4.0, -4.0, 2.0]]),
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=60,
            gui=gui,
        )

        if self.GUI:
            # 隐藏 PyBullet 默认的左右侧边栏和参数面板，打造纯净画幅
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            # 开启高质量阴影（让飞机和红球在雷达网上有投影，极大提升 3D 空间感）
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        self.EPISODE_LEN_SEC = 10 
        self.pid = DSLPIDControl(drone_model=DroneModel.CF2X)

        self.SMOOTH_FACTOR = 0.1
        self.user_input_pos = np.zeros(3)      
        self.current_target_pos = np.zeros(3)  
        
        # 加入 Yaw 控制变量 
        self.target_yaw = 0.0 

        # === 升级：增加 3 维目标相对速度，总计 16 维 ===
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

        self.action_space = gym.spaces.Discrete(11)

        # 建立 NASA BFM 到物理控制量的映射: {动作编号 : (切向过载 n_x, 法向过载 n_n, 滚转角 mu)}
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
        
        # 将动作的最短持续时间（秒）固化为环境属性
        self.action_durations_sec = {
            0:  0.08,  # 匀速直飞（默认状态，给个较短的时间以保持灵活性，约10帧）
            1:  0.2,   # 加速
            2:  0.25,   # 减速
            3:  0.3,   # 跃升
            4:  0.3,   # 俯冲
            9:  0.25,   # 右转
            10: 0.25,   # 左转
            5:  0.4,   # 左转跃升
            7:  0.4,   # 右转跃升
            8:  0.4,   # 左转俯冲
            6:  0.4,   # 右转俯冲
        }
        
        # 用于兼容 manual_control 的标志位
        self.is_manual_mode = False 
        
    def reset(self, seed=None, options=None):
        obs_raw, info = super().reset(seed=seed, options=options)

        # 随机生成运动的“中心锚点” (防止红球钻地，Z轴基础高度调高点)
        self.target_anchor = np.array([
            np.random.uniform(-5.0, 5.0),
            np.random.uniform(-5.0, 5.0),
            np.random.uniform(2.0, 4.0)
        ])
        
        # 随机抽取本局的运动模式 (0:X直线, 1:Y直线, 2:Z上下, 3:水平圆周)
        self.target_mode = np.random.choice([0,1,2,3,4,5])  
        self.target_v = np.zeros(3)

        #初始化各模式的起始状态
        if self.target_mode in [0, 1]:
            # X 轴或 Y 轴水平往复
            self.target_v[self.target_mode] = np.random.choice([-0.8, 0.8])
            self.target_pos = self.target_anchor.copy()
            
        elif self.target_mode == 2:
            # Z 轴垂直往复 (为了不砸地，速度设稍微慢一点 0.6)
            self.target_v[2] = np.random.choice([-0.6, 0.6])
            self.target_pos = self.target_anchor.copy()
            
        elif self.target_mode == 3:
            # 水平面圆周运动
            self.target_angle = np.random.uniform(0, 2 * np.pi) # 随机起始角度
            self.target_omega = np.random.choice([-0.6, 0.6])   # 随机顺时针或逆时针
            self.target_radius = np.random.uniform(1.0, 2.0)    # 随机转圈半径
            
            # 极坐标转直角坐标
            self.target_pos = np.array([
                self.target_anchor[0] + self.target_radius * np.cos(self.target_angle),
                self.target_anchor[1] + self.target_radius * np.sin(self.target_angle),
                self.target_anchor[2]
            ])
            # 切线瞬时速度公式: vx = -R * w * sin(a), vy = R * w * cos(a)
            self.target_v[0] = -self.target_radius * self.target_omega * np.sin(self.target_angle)
            self.target_v[1] =  self.target_radius * self.target_omega * np.cos(self.target_angle)
            self.target_v[2] = 0.0

        # ==========================================
        # 【新增】：模式 4 - 水平 8 字形 (蛇形机动) 起始状态
        # ==========================================
        elif self.target_mode == 4:
            self.target_angle = np.random.uniform(0, 2 * np.pi) 
            self.target_omega = np.random.choice([-0.5, 0.5])   # 角速度稍微调慢一点，蛇形太快很难追
            self.target_radius = np.random.uniform(1.5, 2.5)    # 8字形需要大一点的盘旋半径
            
            self.target_pos = np.array([
                self.target_anchor[0] + self.target_radius * np.cos(self.target_angle),
                self.target_anchor[1] + self.target_radius * np.sin(2.0 * self.target_angle) * 0.8,
                self.target_anchor[2]
            ])
            # 切线初速度导数
            self.target_v[0] = -self.target_radius * self.target_omega * np.sin(self.target_angle)
            self.target_v[1] =  self.target_radius * self.target_omega * 1.6 * np.cos(2.0 * self.target_angle)
            self.target_v[2] = 0.0

        # ==========================================
        # 【新增】：模式 5 - 3D 螺旋逃逸 起始状态
        # ==========================================
        elif self.target_mode == 5:
            self.target_angle = np.random.uniform(0, 2 * np.pi)
            self.target_omega = np.random.choice([-0.5, 0.5])
            self.target_radius = np.random.uniform(1.0, 2.0)
            
            # 安全防沉迷保护：如果随机到的锚点太低，加上正弦振幅后开局可能钻地。所以把锚点至少拔高到 1.5m。
            z_safe_anchor = max(1.5, self.target_anchor[2])
            
            self.target_pos = np.array([
                self.target_anchor[0] + self.target_radius * np.cos(self.target_angle),
                self.target_anchor[1] + self.target_radius * np.sin(self.target_angle),
                z_safe_anchor + np.sin(self.target_angle * 1.5) * 0.5  # 加入 Z 轴的初相位起伏
            ])
            
            # 三维切线初速度
            self.target_v[0] = -self.target_radius * self.target_omega * np.sin(self.target_angle)
            self.target_v[1] =  self.target_radius * self.target_omega * np.cos(self.target_angle)
            self.target_v[2] =  self.target_omega * 1.5 * np.cos(self.target_angle * 1.5) * 0.5

        
        state = self._getDroneStateVector(0)
        self.prev_dist = np.linalg.norm(self.target_pos - state[0:3])

        self.user_input_pos = state[0:3].copy()
        self.current_target_pos = state[0:3].copy()
        self.target_yaw = 0.0 # 每一局重置视角

        # 【新增】：重置文字 ID
        self.hud_text_id = -1  

        if self.GUI:
            p.removeAllUserDebugItems(physicsClientId=self.CLIENT) # 每次重置清除debug line，防止卡顿/内存泄漏
            v_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.08, rgbaColor=[1, 0, 0, 0.8])
            self.target_obj_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=v_id, basePosition=self.target_pos)           
            
            # 【全新视觉】：半透明的近炸引信杀伤圈
            # 设定半径 0.4m，颜色为半透明橙色 (alpha=0.25)
            # 因为没有赋予 CollisionShape，所以它是一个纯视觉的“幽灵球”，不会引发物理碰撞
            # ==========================================
            fuze_v_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.4, rgbaColor=[1, 0.5, 0, 0.25])
            self.fuze_obj_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=fuze_v_id, basePosition=self.target_pos)

            # ==========================================
            # 【全新可视化】：绘制 3D 空间坐标轴与全息网格
            # ==========================================

            z_offset = 0.05  # 视觉悬浮高度

            # 1. 绘制主坐标轴 (X红, Y绿, Z蓝)，跨度扩展到 -10m 到 10m
            p.addUserDebugLine([-10, 0, z_offset],[10, 0, z_offset], [1, 0, 0], 4, 0, physicsClientId=self.CLIENT)
            p.addUserDebugText("X-Axis", [10.5, 0, z_offset], [1, 0, 0], 1.5, physicsClientId=self.CLIENT)
            
            p.addUserDebugLine([0, -10, z_offset], [0, 10, z_offset],[0, 1, 0], 4, 0, physicsClientId=self.CLIENT)
            p.addUserDebugText("Y-Axis", [0, 10.5, z_offset],[0, 1, 0], 1.5, physicsClientId=self.CLIENT)
            
            # Z轴从地面的 offset 开始往上画
            p.addUserDebugLine([0, 0, z_offset],[0, 0, 10], [0, 0.5, 1], 4, 0, physicsClientId=self.CLIENT)
            p.addUserDebugText("Z-Altitude",[0, 0, 10.5], [0, 0.5, 1], 1.5, physicsClientId=self.CLIENT)

            # 2. 添加关键距离刻度线与文字标注 (5m, 10m)
            for val in [-10, -5, 5, 10]:
                # X 轴刻度 (在 X 轴上画小短线，并标字，保持悬浮)
                p.addUserDebugLine([val, -0.2, z_offset], [val, 0.2, z_offset],[1, 0, 0], 2, 0, physicsClientId=self.CLIENT)
                p.addUserDebugText(f"{val}m", [val, 0.5, z_offset],[1, 0.5, 0.5], 1.2, physicsClientId=self.CLIENT)
                
                # Y 轴刻度 (保持悬浮)
                p.addUserDebugLine([-0.2, val, z_offset], [0.2, val, z_offset],[0, 1, 0], 2, 0, physicsClientId=self.CLIENT)
                p.addUserDebugText(f"{val}m", [0.5, val, z_offset], [0.5, 1, 0.5], 1.2, physicsClientId=self.CLIENT)
                
            for val in [5, 10]:
                # Z 轴刻度 (本身在空中，不需要 z_offset)
                p.addUserDebugLine([-0.2, 0, val], [0.2, 0, val],[0, 0.5, 1], 2, 0, physicsClientId=self.CLIENT)
                p.addUserDebugText(f"{val}m", [0.5, 0, val],[0.5, 0.5, 1], 1.2, physicsClientId=self.CLIENT)

            # 3. 绘制底层地平面雷达网格 (Z=0)，范围 -10 到 10，每隔 2 米一条线
            for i in range(-10, 11, 2):
                if i != 0: # 避开中心主坐标轴
                    # 颜色改为深空灰/暗铁色 [0.25, 0.25, 0.25]（或者是暗雷达蓝[0.1, 0.2, 0.3]）
                    grid_color =[0.1, 0.2, 0.3]      
                    p.addUserDebugLine([i, -10, 0.0], [i, 10, 0.0], grid_color, 1.0, 0, physicsClientId=self.CLIENT)
                    p.addUserDebugLine([-10, i, 0.0], [10, i, 0.0], grid_color, 1.0, 0, physicsClientId=self.CLIENT)

        # 【新增】：为高频画线和运镜初始化历史坐标
        # ==========================================
        self.last_draw_pos = state[0:3].copy()
        self.last_target_draw_pos = self.target_pos.copy()
        self.cam_pos = state[0:3].copy()   

        return self._computeObs(), info

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        pilot_quat = p.getQuaternionFromEuler([0, 0, self.target_yaw])
        _, inv_quat = p.invertTransform([0,0,0], pilot_quat)

        # 1. 相对位置映射
        world_rel_pos = self.target_pos - pos
        local_rel_pos, _ = p.multiplyTransforms([0,0,0], inv_quat, world_rel_pos,[0,0,0,1])
        local_rel_pos = np.array(local_rel_pos) 

        # 2. 速度映射
        world_vel = state[10:13]
        local_vel, _ = p.multiplyTransforms([0,0,0], inv_quat, world_vel,[0,0,0,1])
        local_vel = np.array(local_vel)
        
        # 3. 虚拟引导点映射
        world_virt_pos = self.current_target_pos - pos
        local_virt_pos, _ = p.multiplyTransforms([0,0,0], inv_quat, world_virt_pos,[0,0,0,1])
        local_virt_pos = np.array(local_virt_pos)

        # 4. 目标速度映射
        local_target_v, _ = p.multiplyTransforms([0,0,0], inv_quat, self.target_v, [0,0,0,1])
        local_target_v = np.array(local_target_v)

        rpy = state[7:10]   # 自身姿态
        z_height = state[2] # 提取绝对高度
        
        # 将 local_target_v 也拼接到最后，总计 16 个元素
        return np.concatenate([local_rel_pos, local_vel, rpy,[z_height], local_virt_pos, local_target_v]).astype(np.float32)

    def step(self, action):
        state = self._getDroneStateVector(0)
        dt = 1 / self.CTRL_FREQ

        # 解码离散动作
        if isinstance(action, np.ndarray):
            action_int = int(action[0])
        else:
            action_int = int(action)
        n_x, n_n, mu = self.bfm_action_mapping[action_int]
        
        if self.is_manual_mode:
            # 手动模式下，由外部 while 循环控制时间，内部只走 1 帧
            dynamic_frame_skip = 1
        else:
            # 训练模式下，AI 选定一个动作后，强制执行指定的时长！
            duration = self.action_durations_sec.get(action_int, 0.2)
            dynamic_frame_skip = int(duration * self.CTRL_FREQ)
            
        # 初始化累加奖励和终止状态
        total_reward = 0.0
        terminated = False
        truncated = False
        info_pyb = {} # 用于接收底层物理引擎的 info
        
        frame_prev_dist = np.linalg.norm(self.target_pos - state[0:3])

        # 动作重复循环 (Frame Skip)
        repeat = 1 if self.is_manual_mode else dynamic_frame_skip
        for _ in range(repeat):   

            # 将目标的运动逻辑移入内部循环，确保时间流逝与无人机完全同步
            if self.target_mode in [0, 1, 2]:
            # 直线模式 (X/Y/Z)
            # 更新目标位置: 新位置 = 老位置 + 速度 * 时间
                self.target_pos += self.target_v * dt
                axis = self.target_mode
                bounce_limit = 1.0 if axis == 2 else 2.5
            
                # 限制移动范围 
                if abs(self.target_pos[axis] - self.target_anchor[axis]) > bounce_limit:
                    self.target_v[axis] *= -1 # 速度反转
                    # 强行拉回边界内，防止卡墙穿模
                    self.target_pos[axis] = self.target_anchor[axis] + np.sign(self.target_pos[axis] - self.target_anchor[axis]) * bounce_limit
                    
            elif self.target_mode == 3:
                # 圆周模式
                self.target_angle += self.target_omega * dt # 更新极角
                
                # 更新绝对坐标
                self.target_pos[0] = self.target_anchor[0] + self.target_radius * np.cos(self.target_angle)
                self.target_pos[1] = self.target_anchor[1] + self.target_radius * np.sin(self.target_angle)
                
                # 实时更新切线方向的瞬时速度 (这极度重要，无人机会从 local_target_v 里读取这个变化)
                self.target_v[0] = -self.target_radius * self.target_omega * np.sin(self.target_angle)
                self.target_v[1] =  self.target_radius * self.target_omega * np.cos(self.target_angle)
            
            # ==========================================
            # 【新增战术 1】：模式 4 - 水平 8 字形 (蛇形机动 / 连续剪刀战)
            # ==========================================
            elif self.target_mode == 4:
                self.target_angle += self.target_omega * dt
                # 利用李萨如曲线 (Lissajous curve) 生成 8 字形轨迹
                self.target_pos[0] = self.target_anchor[0] + self.target_radius * np.cos(self.target_angle)
                self.target_pos[1] = self.target_anchor[1] + self.target_radius * np.sin(2.0 * self.target_angle) * 0.8
                
                # 速度导数
                self.target_v[0] = -self.target_radius * self.target_omega * np.sin(self.target_angle)
                self.target_v[1] =  self.target_radius * self.target_omega * 1.6 * np.cos(2.0 * self.target_angle)
                self.target_v[2] = 0.0

            # ==========================================
            # 【新增战术 2】：模式 5 - 3D 螺旋逃逸 (水平盘旋 + Z轴起伏)
            # ==========================================
            elif self.target_mode == 5:
                self.target_angle += self.target_omega * dt
                self.target_pos[0] = self.target_anchor[0] + self.target_radius * np.cos(self.target_angle)
                self.target_pos[1] = self.target_anchor[1] + self.target_radius * np.sin(self.target_angle)
                
                # Z 轴缓慢做正弦波起伏波动 (振幅 0.5米)
                self.target_pos[2] = self.target_anchor[2] + np.sin(self.target_angle * 1.5) * 0.5
                
                self.target_v[0] = -self.target_radius * self.target_omega * np.sin(self.target_angle)
                self.target_v[1] =  self.target_radius * self.target_omega * np.cos(self.target_angle)
                self.target_v[2] =  self.target_omega * 1.5 * np.cos(self.target_angle * 1.5) * 0.5
            
            # 防止红球钻入地下 (Z轴托底)
            if self.target_pos[2] < 0.2:
                self.target_pos[2] = 0.2
                if self.target_mode == 2: self.target_v[2] *= -1
        
            # 更新 PyBullet 实体位置
            if self.GUI and self.target_obj_id != -1:
                p.resetBasePositionAndOrientation(self.target_obj_id, self.target_pos,[0, 0, 0, 1])
                p.resetBasePositionAndOrientation(self.fuze_obj_id, self.target_pos,[0, 0, 0, 1])

            # 每一帧必须重新读取最新状态
            current_state = self._getDroneStateVector(0)
            #   - 切向过载 nx 转为前进速度 (默认基础前飞速度1.5m/s，最大3.1m/s)
            vx = 1.5 + n_x * 0.8
            #   - BFM库没有平移侧飞概念，严格遵循空战气动，强制为0
            vy = 0.0 
            #   - 法向过载 nn 与滚转角 mu 的垂直分量转化为爬升/俯冲速度 (平飞时nn*cos(0)=1，抵消重力，vz=0)
            vz = (n_n * np.cos(mu) - 1.0) * 0.15

            if self.user_input_pos[2] <= 1.2 and vz < 0:
                vz = 0.0  # 强行拉平，绝对不允许再往下掉
                
            yaw_rate = n_n * np.sin(mu) * 0.5
            
            target_vel_local = np.array([vx, vy, vz])
            self.target_yaw += yaw_rate * dt
            # 限制 target_yaw 在 [-pi, pi] 之间，防止数值爆炸

            # 生成一个没有俯仰(Pitch)和滚转(Roll)的纯净偏航姿态
            pilot_quat = p.getQuaternionFromEuler([0, 0, self.target_yaw])

            # 将第一人称飞行指令转回世界坐标，用于移动我们的“虚拟目标点”
            vel_world, _ = p.multiplyTransforms([0,0,0], pilot_quat, target_vel_local, [0,0,0,1])
            vel_world = np.array(vel_world)

            self.user_input_pos += vel_world * dt

            if self.user_input_pos[2] < 1.0:
                self.user_input_pos[2] = 1.0   # 虚拟引导点永远不低于 1.0m
            if self.user_input_pos[2] > 8.0:
                self.user_input_pos[2] = 8.0   # 虚拟引导点永远不高于 8.0m
            
            # 给虚拟目标点拴上“物理狗链”
            # 防止动作时间过长时，引导点飞太远导致 PID 崩溃
            dist_carrot = np.linalg.norm(self.user_input_pos - current_state[0:3])
            if dist_carrot > 2.0:
                direction = (self.user_input_pos - current_state[0:3]) / dist_carrot
                self.user_input_pos = current_state[0:3] + direction * 2.0

            self.current_target_pos = self.current_target_pos * (1 - self.SMOOTH_FACTOR) + self.user_input_pos * self.SMOOTH_FACTOR
            
            # 底层 PID 控制
            rpm, _, _ = self.pid.computeControl(
                control_timestep=dt,
                cur_pos=current_state[0:3],       
                cur_quat=current_state[3:7],     
                cur_vel=current_state[10:13],     
                cur_ang_vel=current_state[13:16],
                target_pos=self.current_target_pos, 
                target_vel=vel_world,
                target_rpy=np.array([0, 0, self.target_yaw]) # 将偏航角传递给底层 PID 控制器
            )
            
            obs_raw, _, _, truncated, info_pyb = super().step(rpm.reshape(1, 4))

            if self.GUI:
                # 实现 1:1 真实物理平滑渲染
                time.sleep(1 / self.CTRL_FREQ) 

                if not self.is_manual_mode:
                    keys = p.getKeyboardEvents(physicsClientId=self.CLIENT)
                    if ord('1') in keys and keys[ord('1')] & p.KEY_WAS_TRIGGERED:
                        self.camera_mode = 1
                        print("切换至：智能追尾视角")
                    if ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED:
                        self.camera_mode = 2
                        print("切换至：上帝固定全景视角")
                    if ord('3') in keys and keys[ord('3')] & p.KEY_WAS_TRIGGERED:
                        self.camera_mode = 3
                        print("切换至：目标迎击视角")
                    if ord('4') in keys and keys[ord('4')] & p.KEY_WAS_TRIGGERED:
                        self.camera_mode = 4
                        print("切换至：目标固定广角视角")
                    if ord('5') in keys and keys[ord('5')] & p.KEY_WAS_TRIGGERED:
                        self.camera_mode = 5
                        print("切换至：上帝侧面平视视角")

                cur_draw_pos = current_state[0:3]
                cur_target_draw_pos = self.target_pos.copy()
                dist_cam = np.linalg.norm(cur_draw_pos - cur_target_draw_pos)
                dx = cur_draw_pos[0] - cur_target_draw_pos[0]
                dy = cur_draw_pos[1] - cur_target_draw_pos[1]
                drone_angle = np.degrees(np.arctan2(dy, dx))

                # 覆盖刷新机制 + 动态悬浮跟随
                # ==========================================
                # 提取速度和角度，缩减小数点保留位数，精简缩写
                vel_norm = np.linalg.norm(current_state[10:13])
                # 将多行拼成单行，用 " | " 分隔
                hud_text = f"Dist:{dist_cam:.1f}m | ATA:{drone_angle:.0f}deg | Vel:{vel_norm:.1f}m/s"
                # 获取无人机底层实体 ID (兼容不同版本的 gym-pybullet-drones)
                drone_id = self.DRONE_IDS[0] if hasattr(self, 'DRONE_IDS') else self.drone_ids[0]
                cam_pos =[0, 0, 0.8]
                if self.hud_text_id == -1:
                    self.hud_text_id = p.addUserDebugText(
                        hud_text, cam_pos, textColorRGB=[0, 0, 0], textSize=1.5, 
                        parentObjectUniqueId=drone_id, physicsClientId=self.CLIENT)
                else:
                    self.hud_text_id = p.addUserDebugText(
                        hud_text, cam_pos, textColorRGB=[0, 0, 0], textSize=1.5, 
                        parentObjectUniqueId=drone_id, replaceItemUniqueId=self.hud_text_id, 
                        physicsClientId=self.CLIENT)

                # 60Hz 极其平滑地画线(生命周期可改为 0 用于截图，这里保留 1.5 适合视频)
                p.addUserDebugLine(self.last_draw_pos, cur_draw_pos, [1, 0, 0], 2.5, 3.0, physicsClientId=self.CLIENT)
                p.addUserDebugLine(self.last_target_draw_pos, cur_target_draw_pos, [1, 1, 0], 2.5, 1.5, physicsClientId=self.CLIENT)
                
                self.last_draw_pos = cur_draw_pos
                self.last_target_draw_pos = cur_target_draw_pos

                # 60Hz 电影级平滑运镜
                if not self.is_manual_mode:

                    self.cam_pos = self.cam_pos * 0.95 + cur_draw_pos * 0.05
                    smooth_yaw = np.degrees(self.target_yaw)

                    if self.camera_mode == 1:
                        p.resetDebugVisualizerCamera(2.0, smooth_yaw - 90, -20, self.cam_pos, physicsClientId=self.CLIENT)
                    elif self.camera_mode == 2:
                        p.resetDebugVisualizerCamera(8.0, 0, -89.9,[0, 0, 1.0], physicsClientId=self.CLIENT)
                    elif self.camera_mode == 3:
                        p.resetDebugVisualizerCamera(max(1.5, dist_cam * 0.8), drone_angle - 45, -20, cur_target_draw_pos, physicsClientId=self.CLIENT)
                    elif self.camera_mode == 4:
                        p.resetDebugVisualizerCamera(4.0, 45, -30, cur_target_draw_pos, physicsClientId=self.CLIENT)
                    elif self.camera_mode == 5:
                        anchor_z = self.target_anchor[2]
                        p.resetDebugVisualizerCamera(10.0, 90, -5,[0, 0, anchor_z], physicsClientId=self.CLIENT)
                
            # 优化版 BFM 空战截击奖励函数
            # ==========================================
            
            # 1. 计算相对位置与距离
            cur_pos = current_state[0:3]  # 注意这里用 current_state
            rel_pos = self.target_pos - cur_pos
            dist = np.linalg.norm(rel_pos)
            
            # 计算距离变化率 (接近速度)
            # delta_dist < 0 表示正在靠近，delta_dist > 0 表示正在远离
            frame_delta_dist = dist - frame_prev_dist
            frame_prev_dist = dist  # 为下一帧更新记录

            # 2. 计算机头指向与目标的夹角 (即论文中的 ATA - 天线前置角)
            # 获取无人机当前的姿态四元数转旋转矩阵
            pilot_quat_eval = p.getQuaternionFromEuler([0, 0, self.target_yaw])
            rot_mat = p.getMatrixFromQuaternion(pilot_quat_eval)
            forward_vector = np.array([rot_mat[0], rot_mat[3], rot_mat[6]]) 
                        
            # 归一化目标方向向量
            target_dir = rel_pos / (dist + 1e-6)
            
            # 计算机头与目标的余弦相似度 (-1 到 1，1表示正对目标)
            cos_theta = np.clip(np.dot(forward_vector, target_dir), -1.0, 1.0)
            # 转换为角度差 (弧度, 0 表示正对，pi 表示背对)
            ata_angle = np.arccos(cos_theta) 

            # 引导项：基于势能的靠近奖励 (Progress Reward)
            reward_progress = (self.prev_dist - dist) * 20.0
            self.prev_dist = dist  # 更新历史距离记录

            # 惩罚项 A：追踪偏离惩罚 (Tracking Penalty)
            # 偏离越大扣分越多，完美对准时不扣分 (0 到 -2.0)
            penalty_tracking = - (ata_angle / np.pi) * 2.0 * dt

            # 惩罚项 B：距离惩罚 (Distance Penalty)
            # 离得越远扣分越多，逼迫 AI 拉近距离
            penalty_distance = - (dist * 0.1) * dt

            # 惩罚项 C：时间与动作惩罚 (Time & Action Penalty)
            # 每多活一帧就扣分！持续时间越长的动作扣分越狠，逼迫 AI 速战速决
            penalty_time = -0.1 * dt
            penalty_action = -0.1 * (abs(n_x) + abs(n_n - 1.0) + abs(mu)) * dt

            # 惩罚项 D：危险高度限制 (Height Penalty)
            # 防止 AI 钻地或飞向太空逃避战斗
            penalty_height = 0.0
            if cur_pos[2] < 1.0:
                penalty_height = - (1.0 - cur_pos[2]) * 5.0 * dt # 太低扣分
            elif cur_pos[2] > 5.0:
                penalty_height = - (cur_pos[2] - 5.0) * 5.0 * dt # 太高扣分

            # 结算项 E：稀疏任务奖励 (Terminal Reward)
            reward_terminal = 0.0
            terminated = False
    
            # 情况 1：直接命中 (动能撞击)
            if dist < 0.15:
                reward_terminal = 300.0  # 满分奖励
                terminated = True
                print(f"动能撞击！致命命中！脱靶量: {dist:.2f}m")
                info_pyb["terminal_drone_pos"] = cur_pos.copy()
                info_pyb["terminal_target_pos"] = self.target_pos.copy()
                break

            # 情况 2：擦肩而过，触发近炸引信 (计算脱靶量)
            # 当距离小于杀伤半径(例如 0.4m)，且距离开始变大(frame_delta_dist > 0)时，说明刚刚掠过目标！
            # 此时的 prev_dist 就是本次攻击的“脱靶量 (Miss Distance)”
            elif dist < 0.4 and frame_delta_dist > 0:
                miss_distance = self.prev_dist
                
                # 根据脱靶量计算梯度得分：
                # 脱靶量 0.15m 附近 -> 得分接近 300
                # 脱靶量 0.40m 附近 -> 得分接近 50
                # 公式：基础分50 + 250 * (1 - (脱靶量 - 0.15) / 杀伤区间)
                score_ratio = 1.0 - ((miss_distance - 0.15) / (0.4 - 0.15))
                reward_terminal = 50.0 + 250.0 * score_ratio
                
                terminated = True
                print(f"触发近炸引信！擦肩而过，脱靶量: {miss_distance:.2f}m，获得得分: {reward_terminal:.1f}")
                info_pyb["terminal_drone_pos"] = cur_pos.copy()
                info_pyb["terminal_target_pos"] = self.target_pos.copy()
                break

                
            # 失败惩罚：出界或坠毁
            elif dist > 15.0 or cur_pos[2] < 0.05 or cur_pos[2] > 10.0: 
                reward_terminal = -300.0
                info_pyb["terminal_drone_pos"] = cur_pos.copy()
                info_pyb["terminal_target_pos"] = self.target_pos.copy()
                terminated = True

            if (self.step_counter / self.PYB_FREQ) > self.EPISODE_LEN_SEC:
                truncated = True

            # === 每帧的总奖励全部为负，直至截击成功 ===
            reward = reward_progress + penalty_tracking + penalty_distance + penalty_time + penalty_action + penalty_height + reward_terminal
            total_reward += reward 

            if terminated or truncated:
                break

        # 重新计算循环跳跃结束后的最终环境观测快照
        final_obs = self._computeObs()
        
        # 组装 info 字典（保留底层环境自带的info，同时塞入我们的专家机动动作数据）
        info = info_pyb
        info["BFM_action"] = {
            "action_id": action_int,
            "n_x (g)": n_x, 
            "n_n (g)": n_n, 
            "roll (rad)": round(mu, 2)
        }    
        if terminated or truncated:
            info["is_success"] = bool(reward_terminal >= 50.0)
        
        return final_obs, total_reward, terminated, truncated, info

def plot_mission_trajectory(drone_pos, target_pos, timestamps, episode_id=0):
    """绘制无人机 vs 红球的 3D 轨迹 + 距离曲线"""
    if len(drone_pos) == 0:
        return
    
    drone_pos = np.array(drone_pos)
    target_pos = np.array(target_pos)
    timestamps = np.array(timestamps)
    dist = np.linalg.norm(drone_pos - target_pos, axis=1)

    # === 学术论文全局字体与样式设置 ===
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
    })

    # 高级配色方案 (Tab10)
    C_DRONE = '#1f77b4'  # 沉稳蓝
    C_TARGET = '#d62728' # 学术红
    C_START = '#2ca02c'  # 绿
    C_END = '#9467bd'    # 紫

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Autonomous Interception Trajectory Analysis (Episode {episode_id})', fontsize=18, fontweight='bold', y=0.96)
    
    # 1. 3D 轨迹图（等比例无畸变）
    ax1 = fig.add_subplot(221, projection='3d')
    # 无人机轨迹
    ax1.plot(drone_pos[:,0], drone_pos[:,1], drone_pos[:,2], color=C_DRONE, linewidth=3.0, label='UAV Trajectory')
    # 红球轨迹
    ax1.plot(target_pos[:,0], target_pos[:,1], target_pos[:,2], color=C_TARGET, linewidth=2.5, linestyle='--', label='Target Trajectory')
    # 标注起点和终点
    ax1.scatter(drone_pos[0,0], drone_pos[0,1], drone_pos[0,2], c=C_START, s=100, marker='^', zorder=5, label='Start')
    ax1.scatter(drone_pos[-1,0], drone_pos[-1,1], drone_pos[-1,2], c=C_END, s=100, marker='*', zorder=5, label='Intercept Point')
    ax1.scatter(target_pos[0,0], target_pos[0,1], target_pos[0,2], c=C_START, s=80, marker='o')

    ax1.set_xlabel('X (m)', labelpad=10)
    ax1.set_ylabel('Y (m)', labelpad=10)
    ax1.set_zlabel('Altitude Z (m)', labelpad=10)
    ax1.set_title('3D Spatial Pursuit Geometry')
    ax1.legend(loc='upper left', framealpha=0.9)

    # 强制 3D 等比例显示 (统筹全局，完美防畸变与出界)
    # ==========================================
    # 把无人机和目标的坐标拼在一起，寻找真正的全局边界！
    all_x = np.concatenate([drone_pos[:,0], target_pos[:,0]])
    all_y = np.concatenate([drone_pos[:,1], target_pos[:,1]])
    all_z = np.concatenate([drone_pos[:,2], target_pos[:,2]])

    # 计算全局最大跨度的一半
    max_range = np.array([all_x.max() - all_x.min(), 
                          all_y.max() - all_y.min(), 
                          all_z.max() - all_z.min()]).max() / 2.0
                          
    # 给视角加上 15% 的边缘留白 (Padding)！
    # 防止起点和终点的五角星/大圆点被切掉半个，画幅更加舒展高级
    max_range *= 1.15 

    # 计算真正的全局中心点
    mid_x = (all_x.max() + all_x.min()) * 0.5
    mid_y = (all_y.max() + all_y.min()) * 0.5
    mid_z = (all_z.max() + all_z.min()) * 0.5
    
    # 设定完美的 1:1:1 视野
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # 2. XY 平面投影 + 视线连线 (LOS)
    ax2 = fig.add_subplot(222)
    ax2.plot(drone_pos[:,0], drone_pos[:,1], color=C_DRONE, linewidth=3.0, label='UAV')
    ax2.plot(target_pos[:,0], target_pos[:,1], color=C_TARGET, linewidth=2.5, linestyle='--', label='Target')
    
    # 绘制时间同步的视线角连线 (LOS) 证明拦截战术
    # 严格按照真实的物理时间来画线
    los_interval = 1.0  # 设定：每隔正好 1.0 秒画一条视线 (您可以自由改为 0.5 或 2.0)
    last_los_time = -los_interval  

    for i, t in enumerate(timestamps):
        # 只有当前时间距离上次画线超过了设定的秒数，才画一条线
        if t - last_los_time >= los_interval:
            ax2.plot([drone_pos[i,0], target_pos[i,0]], 
                     [drone_pos[i,1], target_pos[i,1]], 
                     color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
            last_los_time = t
            
    # 【细节加分】：用一根醒目的红色虚线，补上最后“截击瞬间”的最终视线
    ax2.plot([drone_pos[-1,0], target_pos[-1,0]], 
             [drone_pos[-1,1], target_pos[-1,1]], 
             color='#d62728', linestyle=':', linewidth=1.8, alpha=0.8)
    
    ax2.scatter(drone_pos[0,0], drone_pos[0,1], c=C_START, s=120, marker='^', zorder=5)
    ax2.scatter(drone_pos[-1,0], drone_pos[-1,1], c=C_END, s=150, marker='*', zorder=5, label='Intercept')
    ax2.scatter(target_pos[0,0], target_pos[0,1], c=C_START, s=80, marker='o')

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Projection (top view) & Line of Sight (LOS)')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.axis('equal') # 绝对等比例

    # 3. 距离随时间变化曲线 + 杀伤区高亮填充
    ax3 = fig.add_subplot(223)
    ax3.plot(timestamps, dist, color='#2ca02c', linewidth=3.0, label='Distance To Target')
    
    # 标注最小距离点
    min_idx = np.argmin(dist)
    min_t, min_d = timestamps[min_idx], dist[min_idx]
    ax3.scatter(min_t, min_d, color='red', s=100, zorder=5)
    ax3.annotate(f'min_dis: {min_d:.2f}m @ {min_t:.1f}s', xy=(min_t, min_d), 
                 xytext=(10, 20), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    # 高亮杀伤区域 (色块填充比画线更具学术感)
    ax3.axhline(y=0.4, color='#ff7f0e', linestyle='--', linewidth=1.5, label='Fuze Radius (0.4m)')
    ax3.fill_between(timestamps, 0, 0.4, color='#ff7f0e', alpha=0.15)
    
    ax3.axhline(y=0.15, color='#d62728', linestyle=':', linewidth=1.5, label='Kinetic Hit (0.15m)')
    ax3.fill_between(timestamps, 0, 0.15, color='#d62728', alpha=0.2)

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Distance (m)')
    ax3.set_title('Distance over Time')
    ax3.set_ylim(bottom=0) # 距离肯定大于0
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.grid(True, linestyle='--', alpha=0.7)

    # 4. 高度随时间对比曲线
    ax4 = fig.add_subplot(224)
    ax4.plot(timestamps, drone_pos[:,2], color=C_DRONE, linewidth=2.5, label='UAV Altitude')
    ax4.plot(timestamps, target_pos[:,2], color=C_TARGET, linewidth=2.5, linestyle='--', label='Target Altitude')
    
    # 填充两者的绝对高度差，直观展示 Z 轴追击过程
    ax4.fill_between(timestamps, drone_pos[:,2], target_pos[:,2], color='gray', alpha=0.15, label='Altitude Gap')

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Z Altitude (m)')
    ax4.set_title('Altitude Pursuit Profile')
    ax4.legend(loc='best', framealpha=0.9)
    ax4.grid(True, linestyle='--', alpha=0.7)

    # 使用和 3D 图 (ax1) 完全相同的 Z 轴上下限，
    # 让高度差在视觉上与 X、Y 轴的位移比例绝对统一！
    # ==========================================
    ax4.set_ylim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9) # 给主标题留出空间

    os.makedirs("results_mission", exist_ok=True)
    save_path = f"results_mission/episode_{episode_id}_trajectory.png"
    plt.savefig(save_path, dpi=350, bbox_inches='tight')
    plt.show()
    
    print(f"已保存 Episode {episode_id} 的轨迹图 → results_mission/episode_{episode_id}_trajectory.png")
    print(f"时长: {timestamps[-1]:.2f}s | 脱靶量: {min_d:.3f}m")

# ---------------------------------------------------------
# 训练与测试设置 
# ---------------------------------------------------------

MODEL_PATH = "drone_vision_v8"
VEC_NORM_PATH = "vec_normalize_vision_v8.pkl"


def train():
    env = DummyVecEnv([lambda: Monitor(DronePIDEnv(gui=False))])
    
    if os.path.exists(VEC_NORM_PATH):
        print("发现旧的归一化文件，加载中...")
        env = VecNormalize.load(VEC_NORM_PATH, env)
        env.training = True     
        env.norm_reward = True  
    else:
        print("未找到归一化文件，新建中...")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"发现旧模型 {MODEL_PATH}.zip，正在加载并进行断点续训！")
        model = PPO.load(MODEL_PATH, env=env) 
    else:
        print("未找到旧模型，从头开始初始化训练...")
        policy_kwargs = dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]))
        model = PPO("MlpPolicy", env, verbose=1, batch_size=256, 
                    learning_rate=1e-4, n_steps=1024, 
                    ent_coef=0.001,
                    policy_kwargs=policy_kwargs, tensorboard_log="./ppo_logs/")

    print("开始导航训练 (按下 Ctrl+C 可以安全提前停止并保存)...")
    try:
        model.learn(total_timesteps=100_000, reset_num_timesteps=False,tb_log_name="run_4.15_BFM") 
    except KeyboardInterrupt:
        print("\n 检测到中止信号，正在保存模型...")
    finally:
        model.save(MODEL_PATH)
        env.save(VEC_NORM_PATH) 
        env.close()
        print("模型与环境参数已成功保存！")

def test():
    print("==================================")
    print("开始演示！")
    print("运镜控制说明 (请确保鼠标点击激活了 PyBullet 物理窗口)：")
    print("  按键 [1] : 智能追尾视角 (跟随无人机)")
    print("  按键 [2] : 上帝固定视角 （俯视全景）")
    print("  按键 [3] : 目标迎击视角 ")
    print("  按键 [4] : 目标固定广角 (以红球为中心，拉远固定视角)")
    print("  按键 [5] : 上帝侧面平视视角 (固定在场地一侧，看高度拉扯)")
    print("==================================")

    env = DummyVecEnv([lambda: DronePIDEnv(gui=True)])
    
    if not os.path.exists(VEC_NORM_PATH):
        print("错误: 找不到 vec_normalize.pkl，请先执行 train()！")
        return
        
    env = VecNormalize.load(VEC_NORM_PATH, env)
    env.training = False 
    env.norm_reward = False

    model = PPO.load(MODEL_PATH)
    base_env = env.venv.envs[0]

    # 新增：初始化 Logger（单无人机）
    logger = Logger(logging_freq_hz=base_env.CTRL_FREQ,   # 通常是 60
                    num_drones=1,
                    output_folder="results_mission",      # 可自定义文件夹
                    colab=False)
    
    # 新增：用于记录红球位置（每帧都记）
    target_history = []      # 红球位置列表 [(x,y,z), ...]
    drone_history = []       # 无人机位置列表（可选，与 Logger 互补）
    timestamps = []          # 时间戳
    obs = env.reset()
    episode_count = 0
    episode_step = 0
    current_episode_id = 1

    while episode_count < 10: 
        action, _ = model.predict(obs, deterministic=True) # AI 思考并输出动作

        real_cur_pos = base_env._getDroneStateVector(0)[0:3]
        print(f"AI 打算执行动作: [{action[0] if isinstance(action, np.ndarray) else action}], 执行前绝对高度: {real_cur_pos[2]:.2f}m")
    
        obs, reward, done, info = env.step(action)         #环境执行动作
        
        # ================== 日志记录 ==================
        current_time = base_env.step_counter / base_env.PYB_FREQ   # 使用环境内部的真实物理时间
        raw_state = base_env._getDroneStateVector(0)
        
        logger_state = np.zeros(20)
        logger_state[0:3]  = raw_state[0:3]
        logger_state[3:7]  = raw_state[3:7] if len(raw_state) > 7 else np.zeros(4)
        logger_state[7:10] = raw_state[7:10] if len(raw_state) > 10 else np.zeros(3)
        logger_state[10:13]= raw_state[10:13] if len(raw_state) > 13 else np.zeros(3)
        logger_state[13:16]= raw_state[13:16] if len(raw_state) > 16 else np.zeros(3)
        logger_state[16:20]= np.zeros(4)   # RPM，可后续改进为真实值

        logger.log(drone=0, 
                timestamp=episode_step / base_env.CTRL_FREQ,
                state=logger_state,
                control=np.zeros(12))
        target_history.append(base_env.target_pos.copy())
        drone_history.append(raw_state[0:3].copy())
        timestamps.append(current_time)

        episode_step += 1
      
        if done[0]:
            episode_count += 1
            print(f"第 {episode_count} 轮测试结束！")

            if len(drone_history) > 0:    # 移除最后一个被 auto-reset 污染的点,避免与reset点连线
                drone_history.pop()
                target_history.pop()
                timestamps.pop()

            last_info = info[0]
            if "terminal_drone_pos" in last_info:
                # 把真正发生碰撞/引信起爆那一帧的坐标，硬塞到图表数据里
                drone_history.append(last_info["terminal_drone_pos"])
                target_history.append(last_info["terminal_target_pos"])
                
                # 稍微加一点时间 (代表这是在宏观动作内部发生的事)
                dt_mock = 0.1 
                timestamps.append(timestamps[-1] + dt_mock if len(timestamps)>0 else dt_mock)

            if len(drone_history) > 10:   # 防止空episode画图
                plot_mission_trajectory(drone_history, target_history, timestamps, current_episode_id)
            
            # 重置本轮记录列表
            target_history = []
            drone_history = []
            timestamps = []
            episode_step = 0   # 重置计步器
            current_episode_id += 1
            obs = env.reset()  # 注意：reset 后要重新取 base_env

    # 全部测试结束后统一画图
    # logger.plot()
    env.close()

def manual_control():
    print("==================================")
    print("开启第一人称离散动作 (BFM) 手动驾驶模式！")
    print("操作说明 (经典飞行器键位，点按生效)：")
    print("  [1] : 匀速直飞 (松开按键默认恢复此状态)")
    print("  [2] : 加速直飞    [3] : 减速直飞/悬停")
    print("  [W] : 跃升        [S] : 俯冲")
    print("  [A] : 左平转      [D] : 右平转")
    print("  [Q] : 左转跃升    [E] : 右转跃升")
    print("  [Z] : 左转俯冲    [C] : 右转俯冲")
    print("  [R] : 重置环境    [ESC]: 退出")
    print("==================================")

    # ====================== 新增：动作优先级与可调持续时间 ======================
    # 优先级（数字越大优先级越高）：组合机动 > 姿态机动 > 速度机动
    PRIORITY = {
        0:  0,   # 默认直飞
        1:  1,   # 加速
        2:  1,   # 减速
        3:  2,   # 跃升
        4:  2,   # 俯冲
        9:  2,   # 右平转
        10: 2,   # 左平转
        5:  3,   # 左转跃升
        7:  3,   # 右转跃升
        8:  3,   # 左转俯冲
        6:  3,   # 右转俯冲
    }

    # 默认最小持续时间（单位：秒）——基于真实空战 BFM 经验值
    # 你可以在这里自由修改每个动作的具体时长
    MIN_DURATIONS = {
        1:  0.2,   # 加速
        2:  0.25,   # 减速
        3:  0.3,   # 跃升
        4:  0.3,   # 俯冲
        9:  0.25,   # 右转
        10: 0.25,   # 左转
        5:  0.4,   # 左转跃升
        7:  0.4,   # 右转跃升
        8:  0.4,   # 左转俯冲
        6:  0.4,   # 右转俯冲
    }

    # 动作 → 对应键盘码（用于“按住延长”功能）
    ACTION_TO_KEY = {
        1: ord('2'), 2: ord('3'),
        3: ord('w'), 4: ord('s'),
        5: ord('q'), 6: ord('c'),
        7: ord('e'), 8: ord('z'),
        9: ord('d'), 10: ord('a'),
    }
    # =========================================================================

    env = DronePIDEnv(gui=True)
    env.is_manual_mode = True
    env.EPISODE_LEN_SEC = 60

    obs, info = env.reset()
    prev_pos = env._getDroneStateVector(0)[0:3]
    cam_pos = prev_pos.copy()

    # 新增持久变量
    current_action = 0
    action_remaining_time = 0.0
    last_time = time.time()

    while True:
        keys = p.getKeyboardEvents()
        dt = time.time() - last_time
        last_time = time.time()

        # 更新剩余强制执行时间
        if action_remaining_time > 0:
            action_remaining_time -= dt
            if action_remaining_time < 0:
                action_remaining_time = 0

        # 退出与重置
        if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
            break

        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            obs, info = env.reset()
            prev_pos = env._getDroneStateVector(0)[0:3]
            cam_pos = prev_pos.copy()
            current_action = 0
            action_remaining_time = 0.0
            continue

        # === 收集本次按下的新动作（使用 WAS_TRIGGERED，支持同时按多个键）===
        triggered_actions = []
        if ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED:
            triggered_actions.append(1)
        if ord('3') in keys and keys[ord('3')] & p.KEY_WAS_TRIGGERED:
            triggered_actions.append(2)
        if ord('w') in keys and keys[ord('w')] & p.KEY_WAS_TRIGGERED:
            triggered_actions.append(3)
        if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
            triggered_actions.append(4)
        if ord('a') in keys and keys[ord('a')] & p.KEY_WAS_TRIGGERED:
            triggered_actions.append(10)
        if ord('d') in keys and keys[ord('d')] & p.KEY_WAS_TRIGGERED:
            triggered_actions.append(9)
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            triggered_actions.append(5)
        if ord('e') in keys and keys[ord('e')] & p.KEY_WAS_TRIGGERED:
            triggered_actions.append(7)
        if ord('z') in keys and keys[ord('z')] & p.KEY_WAS_TRIGGERED:
            triggered_actions.append(8)
        if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
            triggered_actions.append(6)

        # 如果有新按键触发，按优先级选择最高的一个
        if triggered_actions:
            new_action = max(triggered_actions, key=lambda a: PRIORITY.get(a, 0))
            current_action = new_action
            action_remaining_time = MIN_DURATIONS.get(new_action, 0.5)

        # 按住延长机制：如果当前动作对应的键仍然按住，就把剩余时间刷新到至少默认时长
        if current_action != 0:
            key_code = ACTION_TO_KEY.get(current_action)
            if key_code and key_code in keys and keys[key_code] & p.KEY_IS_DOWN:
                action_remaining_time = max(action_remaining_time,
                                            MIN_DURATIONS.get(current_action, 0.5))

        # 决定本次真正要执行的动作
        if action_remaining_time > 0:
            action = current_action
        else:
            action = 0   # 默认匀速直飞

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(np.array([action]))

        # 绘制轨迹、目标点、相机（保持原逻辑）
        cur_pos = env._getDroneStateVector(0)[0:3]
        p.addUserDebugLine(prev_pos, cur_pos, [0, 1, 1], 2.5, 1.5, physicsClientId=env.CLIENT)
        prev_pos = cur_pos

        carrot = env.current_target_pos
        p.addUserDebugLine(carrot-[0.1,0,0], carrot+[0.1,0,0], [1,0,1], 2, 0.05, physicsClientId=env.CLIENT)
        p.addUserDebugLine(carrot-[0,0.1,0], carrot+[0,0.1,0], [1,0,1], 2, 0.05, physicsClientId=env.CLIENT)

        cam_pos = cam_pos * 0.8 + cur_pos * 0.2
        smooth_yaw = np.degrees(env.target_yaw)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5, cameraYaw=smooth_yaw - 90,
            cameraPitch=-15, cameraTargetPosition=cam_pos
        )

        if terminated or truncated:
            obs, info = env.reset()
            prev_pos = env._getDroneStateVector(0)[0:3]
            cam_pos = prev_pos.copy()
            current_action = 0
            action_remaining_time = 0.0

    env.close()

if __name__ == "__main__":
    # manual_control()
    # train()
    test()
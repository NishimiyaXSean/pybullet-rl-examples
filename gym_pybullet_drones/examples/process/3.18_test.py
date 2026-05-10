import os
import time
import numpy as np
import pybullet as p
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class DronePIDEnv(CtrlAviary):
    def __init__(self, gui=False):
        self.target_pos = np.array([1.0, 1.0, 1.0])
        self.target_obj_id = -1
        self.prev_dist = 0.0
        
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=np.array([[0, 0, 1.0]]),
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=30,
            gui=gui,
        )
        self.EPISODE_LEN_SEC = 20 
        self.pid = DSLPIDControl(drone_model=DroneModel.CF2X)

        self.SMOOTH_FACTOR = 0.1
        self.user_input_pos = np.zeros(3)      
        self.current_target_pos = np.zeros(3)  
        self.target_yaw = 0.0 

        self.target_anchor = np.zeros(3) 
        self.target_v = np.zeros(3)      
        self.target_mode = 0        
        self.target_angle = 0.0     
        self.target_omega = 0.0     
        self.target_radius = 1.5    

        self.ray_line_ids =[]

        # ==========================================
        # 🌟 核心升级 1：7x7 狙击级高密度雷达 + FPV 仰角
        # ==========================================
        self.num_rays_h = 7
        self.num_rays_v = 7
        self.num_rays = self.num_rays_h * self.num_rays_v # 共 49 根射线
        self.ray_len = 5.0
        self.ray_line_ids =[]
        self.ray_ends_local =[]
        
        # 1. 水平视角：-45度 到 +45度 (超宽视野，捕获动态侧移)
        angles_y = np.linspace(np.deg2rad(-45), np.deg2rad(45), self.num_rays_h)
        
        # 2. 垂直视角：非对称设计！-15度(微俯视) 到 +30度(仰视防低头)
        angles_z = np.linspace(np.deg2rad(-15), np.deg2rad(30), self.num_rays_v)

        for az in angles_z:
            for ay in angles_y:
                # 极坐标直接转直角坐标 (机头正前方为 X 轴)
                end_x = self.ray_len * np.cos(ay) * np.cos(az)
                end_y = self.ray_len * np.sin(ay) * np.cos(az)
                end_z = self.ray_len * np.sin(az)
                
                self.ray_ends_local.append([end_x, end_y, end_z])
                
        self.ray_ends_local = np.array(self.ray_ends_local)

        # 观察空间：基础状态(13维) + 雷达深度阵列(49维) = 62 维！
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(62,), dtype=np.float32)
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # 提前清空列表，保证 _computeObs() 执行时会重新创建 25 根新射线
        self.ray_line_ids =[]
        obs_raw, info = super().reset(seed=seed, options=options)
        
        self.target_anchor = np.array([
            np.random.uniform(-5.0, 5.0),
            np.random.uniform(-5.0, 5.0),
            np.random.uniform(1.0, 4.0)
        ])
        
        self.target_mode = np.random.randint(0, 4)
        self.target_v = np.zeros(3)
        
        if self.target_mode in[0, 1]:
            self.target_v[self.target_mode] = np.random.choice([-0.8, 0.8])
            self.target_pos = self.target_anchor.copy() 
        elif self.target_mode == 2:
            self.target_v[2] = np.random.choice([-0.6, 0.6])
            self.target_pos = self.target_anchor.copy()
        elif self.target_mode == 3:
            self.target_angle = np.random.uniform(0, 2 * np.pi) 
            self.target_omega = np.random.choice([-0.8, 0.8])   
            self.target_radius = np.random.uniform(1.0, 2.0)    
            
            self.target_pos = np.array([
                self.target_anchor[0] + self.target_radius * np.cos(self.target_angle),
                self.target_anchor[1] + self.target_radius * np.sin(self.target_angle),
                self.target_anchor[2]
            ])
            self.target_v[0] = -self.target_radius * self.target_omega * np.sin(self.target_angle)
            self.target_v[1] =  self.target_radius * self.target_omega * np.cos(self.target_angle)
            self.target_v[2] = 0.0

        state = self._getDroneStateVector(0)
        self.prev_dist = np.linalg.norm(self.target_pos - state[0:3])

        self.user_input_pos = state[0:3].copy()
        self.current_target_pos = state[0:3].copy()
        self.target_yaw = 0.0 

        # 1. 创建可见的视觉外衣
        v_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.08, rgbaColor=[1, 0, 0, 0.8])
        # 2. 创建雷达可以扫到的实体碰撞箱
        c_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.08)
        # 3. 把它们组合成一个实体，注入到物理世界中
        self.target_obj_id = p.createMultiBody(
            baseMass=0,                    # 质量为0表示它是不会受重力掉落的“上帝物体”
            baseCollisionShapeIndex=c_id,  # 加入碰撞箱
            baseVisualShapeIndex=v_id, 
            basePosition=self.target_pos
        )           
        
        return self._computeObs(), info

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        quat = state[3:7] 

        # 基础自身状态
        world_rel_pos = self.target_pos - pos
        _, inv_quat = p.invertTransform([0,0,0], quat)
        local_rel_pos, _ = p.multiplyTransforms([0,0,0], inv_quat, world_rel_pos,[0,0,0,1])
        local_rel_pos = np.array(local_rel_pos) 
        
        world_vel = state[10:13]
        local_vel, _ = p.multiplyTransforms([0,0,0], inv_quat, world_vel, [0,0,0,1])
        world_virt_pos = self.current_target_pos - pos
        local_virt_pos, _ = p.multiplyTransforms([0,0,0], inv_quat, world_virt_pos,[0,0,0,1])

        rpy = state[7:10]
        z_height = state[2] 

        # ==========================================
        # 🌟 核心升级 2：发射激光雷达并获取深度数据
        # ==========================================
        # 将本地射线通过旋转矩阵转换到世界坐标系
        rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        ray_ends_world = pos + np.dot(self.ray_ends_local, rot_mat.T)
        ray_starts_world = np.tile(pos, (self.num_rays, 1))

        # 批量发射物理射线
        ray_results = p.rayTestBatch(ray_starts_world.tolist(), ray_ends_world.tolist())
        
        lidar_depths =[]
        target_hits = 0 # 统计有多少根射线真正打中了红球！

        for i in range(self.num_rays):
            hit_obj_id = ray_results[i][0] # 获取射线打中了什么物体的 ID
            hit_frac = ray_results[i][2]   # 获取射线的深度比例
            
            lidar_depths.append(hit_frac)
            
            # 敌我识别：如果打中的 ID 刚好是红球的 ID，说明火控雷达锁定了！
            if hit_obj_id == self.target_obj_id:
                target_hits += 1
                color =[1, 0, 0] # 打中红球，画红线 (锁定)
            else:
                color =[0, 1, 0] # 没打中或打中地面，画绿线 (安全)

            if self.GUI:
                end_pt = ray_starts_world[i] + (ray_ends_world[i] - ray_starts_world[i]) * hit_frac
                
                if len(self.ray_line_ids) < self.num_rays:
                    line_id = p.addUserDebugLine(ray_starts_world[i], end_pt, color, 1.5, 0, physicsClientId=self.CLIENT)
                    self.ray_line_ids.append(line_id)
                else:
                    p.addUserDebugLine(ray_starts_world[i], end_pt, color, 1.5, 0, replaceItemUniqueId=self.ray_line_ids[i], physicsClientId=self.CLIENT)

        # 把这帧雷达锁定红球的数量存下来，留给 step() 发放奖励
        self.current_lidar_hits = target_hits
        
        lidar_depths = np.array(lidar_depths, dtype=np.float32)        
        
        # 拼接返回观察值
        return np.concatenate([local_rel_pos, local_vel, rpy, [z_height], local_virt_pos, lidar_depths]).astype(np.float32)

    def step(self, action):
        state = self._getDroneStateVector(0)
        dt = 1 / self.CTRL_FREQ
        
        # 目标物理运动推演
        if self.target_mode in [0, 1, 2]:
            self.target_pos += self.target_v * dt
            axis = self.target_mode
            if abs(self.target_pos[axis] - self.target_anchor[axis]) > 2.0:
                self.target_v[axis] *= -1 
                self.target_pos[axis] = self.target_anchor[axis] + np.sign(self.target_pos[axis] - self.target_anchor[axis]) * 2.0
                
        elif self.target_mode == 3:
            self.target_angle += self.target_omega * dt 
            self.target_pos[0] = self.target_anchor[0] + self.target_radius * np.cos(self.target_angle)
            self.target_pos[1] = self.target_anchor[1] + self.target_radius * np.sin(self.target_angle)
            self.target_v[0] = -self.target_radius * self.target_omega * np.sin(self.target_angle)
            self.target_v[1] =  self.target_radius * self.target_omega * np.cos(self.target_angle)
            
        if self.target_pos[2] < 0.2:
            self.target_pos[2] = 0.2
            if self.target_mode == 2: self.target_v[2] *= -1

        if self.target_obj_id != -1:
            p.resetBasePositionAndOrientation(self.target_obj_id, self.target_pos,[0, 0, 0, 1])

        # 无人机飞行控制
        quat = state[3:7]
        target_vel_local = action[0:3] * 2.0 
        yaw_rate = action[3] * 1.5 
        self.target_yaw += yaw_rate * dt
        
        vel_world, _ = p.multiplyTransforms([0,0,0], quat, target_vel_local, [0,0,0,1])
        self.user_input_pos += np.array(vel_world) * dt
        if self.user_input_pos[2] < 0.1: self.user_input_pos[2] = 0.1
        
        self.current_target_pos = self.current_target_pos * (1 - self.SMOOTH_FACTOR) + self.user_input_pos * self.SMOOTH_FACTOR
        
        rpm, _, _ = self.pid.computeControl(
            control_timestep=dt, cur_pos=state[0:3], cur_quat=state[3:7],
            cur_vel=state[10:13], cur_ang_vel=state[13:16],
            target_pos=self.current_target_pos, target_vel=np.zeros(3),
            target_rpy=np.array([0, 0, self.target_yaw]) 
        )
        
        obs_raw, _, _, truncated, info = super().step(rpm.reshape(1, 4))
        
        # --- 奖励函数 (雷达火控 + 末端绝杀模式) ---
        new_state = self._getDroneStateVector(0)
        new_dist = np.linalg.norm(self.target_pos - new_state[0:3])
        
        reward = 0.0

        # 模块 1：绝对驱动力 (进度)
        progress = self.prev_dist - new_dist
        reward_progress = progress * 150.0 
        
        # 模块 2：雷达火控锁定奖励 (Active Radar Seeker)
        # 只要有射线打中红球，每根射线给 0.02 分。
        reward_radar = self.current_lidar_hits * 0.02

        # 模块 3：基础素养与姿态 (中远距离生效)
        reward_vision = 0.0
        reward_safety = 0.0
        reward_smoothness = -0.02 * np.linalg.norm(action)**2 
        reward_time = -0.05   

        # 绝杀区判定 (Kill Zone)：大于 0.6 米才考核姿态，小于 0.6 米全放开！
        if new_dist > 0.6:
            _, inv_quat = p.invertTransform([0,0,0], new_state[3:7])
            local_rel, _ = p.multiplyTransforms([0,0,0], inv_quat, self.target_pos - new_state[0:3],[0,0,0,1])
            local_xy_dist = np.linalg.norm(local_rel[0:2])
            
            if local_xy_dist > 0.2:
                cos_yaw_angle = np.clip(local_rel[0] / local_xy_dist, -1.0, 1.0)
                reward_vision = -0.1 * (1.0 - cos_yaw_angle) 
                
            # 危险倾角惩罚 (中远距离保命)
            roll, pitch = new_state[7:9]
            if abs(roll) > 1.0 or abs(pitch) > 1.0:       
                reward_safety = -2.0
        else:
            # 进入 0.6 米绝杀区：取消视觉和姿态惩罚，并给予高额的距离引力！
            # 让它毫不犹豫、张牙舞爪地撞上去！
            reward_proximity = 0.2 * ((0.6 - new_dist) / 0.6)
            reward += reward_proximity

        # 模块 4：引信触发 (Terminal)
        reward_terminal = 0.0
        terminated = False
        if new_dist < 0.15:
            reward_terminal = 200.0  
            terminated = True
        elif new_state[2] < 0.1 or new_dist > 15.0:
            reward_terminal = -50.0  
            terminated = True

        # === 汇总总分 ===
        reward += (reward_progress + reward_radar + reward_vision + 
                   reward_smoothness + reward_time + reward_safety + reward_terminal)
        
        self.prev_dist = new_dist
        if (self.step_counter / self.PYB_FREQ) > 20.0: truncated = True
            
        return self._computeObs(), reward, terminated, truncated, info

# ---------------------------------------------------------
# 训练与测试设置 
# ---------------------------------------------------------

MODEL_PATH = "drone_lidar_v1"
VEC_NORM_PATH = "vec_normalize_lidar_v1.pkl"

def create_env(gui=False):
    # ==========================================
    # 🌟 核心升级 3：应用帧堆叠 (Frame Stack) 包装器
    # ==========================================
    env = DummyVecEnv([lambda: DronePIDEnv(gui=gui)])
    # 将最近 4 帧的数据堆叠在一起，赋予 AI 感知速度的能力！
    env = VecFrameStack(env, n_stack=4)
    return env

def train():
    env = create_env(gui=False)
    
    if os.path.exists(VEC_NORM_PATH):
        print("✅ 加载归一化文件...")
        env = VecNormalize.load(VEC_NORM_PATH, env)
        env.training = True     
        env.norm_reward = True  
    else:
        print("🆕 新建归一化文件...")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"✅ 断点续训模型 {MODEL_PATH}.zip")
        model = PPO.load(MODEL_PATH, env=env) 
    else:
        print("🆕 从头开始初始化 LiDAR 训练...")
        # 因为输入特征增加到了 35x4 = 140维，稍微扩大神经网络容量
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        model = PPO("MlpPolicy", env, verbose=1, batch_size=256, 
                    learning_rate=3e-4, n_steps=1024, 
                    policy_kwargs=policy_kwargs, tensorboard_log="./ppo_logs/")

    try:
        model.learn(total_timesteps=800_000, reset_num_timesteps=False) 
    except KeyboardInterrupt:
        print("\n🛑 提前保存模型...")
    finally:
        model.save(MODEL_PATH)
        env.save(VEC_NORM_PATH) 
        env.close()

def test():
    env = create_env(gui=True)
    env = VecNormalize.load(VEC_NORM_PATH, env)
    env.training = False 
    env.norm_reward = False

    model = PPO.load(MODEL_PATH)
    # VecFrameStack 和 DummyVecEnv 嵌套了两层，所以要剥开两层获取到底层环境
    base_env = env.venv.envs[0]
    
    obs = env.reset()

    prev_pos = base_env._getDroneStateVector(0)[0:3] # 记录无人机旧位置
    prev_target_pos = base_env.target_pos.copy()     # 记录红球旧位置
    cam_pos = prev_pos.copy()                        # 虚拟云台位置

    camera_mode = 1
    episode_count = 0

    while episode_count < 20: 
        # === 键盘监听：视角切换 ===
        keys = p.getKeyboardEvents(physicsClientId=base_env.CLIENT)
        if ord('1') in keys and keys[ord('1')] & p.KEY_WAS_TRIGGERED:
            camera_mode = 1
            print("切换至：迎击视角")
        if ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED:
            camera_mode = 2
            print("切换至：目标固定广角视角")

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        cur_pos = base_env._getDroneStateVector(0)[0:3]
        cur_target_pos = base_env.target_pos.copy()

        if not done[0]:
            # === 画线 1：无人机飞行轨迹 (红色) ===
            p.addUserDebugLine(
                lineFromXYZ=prev_pos, lineToXYZ=cur_pos, 
                lineColorRGB=[1, 0, 0], lineWidth=2.5, lifeTime=1.5, 
                physicsClientId=base_env.CLIENT
            )
            prev_pos = cur_pos
            
            # === 画线 2：红球运动轨迹 (黄色) ===
            p.addUserDebugLine(
                lineFromXYZ=prev_target_pos, lineToXYZ=cur_target_pos, 
                lineColorRGB=[1, 1, 0], lineWidth=2.5, lifeTime=1.5, 
                physicsClientId=base_env.CLIENT
            )
            prev_target_pos = cur_target_pos
        
            if camera_mode == 1:
                # 默认模式：迎击视角
                cur_pos = base_env._getDroneStateVector(0)[0:3]
                cur_target_pos = base_env.target_pos.copy()
                
                dist = np.linalg.norm(cur_pos - cur_target_pos)
                dx = cur_pos[0] - cur_target_pos[0]
                dy = cur_pos[1] - cur_target_pos[1]
                drone_angle = np.degrees(np.arctan2(dy, dx))

                p.resetDebugVisualizerCamera(max(1.5, dist * 0.8), drone_angle - 45, -20, cur_target_pos, physicsClientId=base_env.CLIENT)

            elif camera_mode == 2:
                # 模式 2：目标固定广角视角 (红球中心，机位锁定在固定角度)
                # 无论红球怎么跑，镜头始终在它右上方的固定 4 米处俯视它，视野极其开阔！
                p.resetDebugVisualizerCamera(
                    cameraDistance=4.0,          # 距离拉远，涵盖整个拦截空域
                    cameraYaw=45,                # 固定偏航角
                    cameraPitch=-30,             # 固定俯仰角
                    cameraTargetPosition=cur_target_pos, # 焦点锁死红球
                    physicsClientId=base_env.CLIENT
                )
            
        time.sleep(1/60) 
        if done[0]:
            episode_count += 1
            print(f"第 {episode_count} 轮测试结束！")

            prev_pos = base_env._getDroneStateVector(0)[0:3]
            prev_target_pos = base_env.target_pos.copy()
            cam_pos = prev_pos.copy() # 重置云台位置

    env.close()
    
def manual_control():
    print("==================================")
    print("开启带激光雷达 (LiDAR) 的手动驾驶模式！")
    print("操作说明 (请确保鼠标点击激活了 PyBullet 仿真窗口)：")
    print("  [U] / [J] : 前进 / 后退 (本地 X 轴)")
    print("  [H] / [K] : 转向左 / 转向右 (Yaw 偏航控制!)")
    print("  [↑] / [↓] : 升高 / 降低 (Z轴)")
    print("  [←] / [→] : 侧飞左 / 侧飞右 (本地 Y 轴)")
    print("  [R]       : 重置环境 | [Q] 退出")
    print("==================================")

    # 注意：手动体验物理特性不需要 VecFrameStack 包装，直接实例化即可
    env = DronePIDEnv(gui=True)
    obs, info = env.reset()
    
    # 初始化虚拟云台
    cam_pos = env._getDroneStateVector(0)[0:3].copy()

    while True:
        action = np.zeros(4, dtype=np.float32) 
        keys = p.getKeyboardEvents()
        
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED: break
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            obs, info = env.reset()
            cam_pos = env._getDroneStateVector(0)[0:3].copy()
            continue

        # FPV 第一人称操作映射
        if ord('u') in keys and keys[ord('u')] & p.KEY_IS_DOWN: action[0] = 1.0
        if ord('j') in keys and keys[ord('j')] & p.KEY_IS_DOWN: action[0] = -1.0
        
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN: action[1] = 1.0
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN: action[1] = -1.0
            
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN: action[2] = 1.0
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN: action[2] = -1.0
            
        if ord('h') in keys and keys[ord('h')] & p.KEY_IS_DOWN: action[3] = 1.0 # 视角左转
        if ord('k') in keys and keys[ord('k')] & p.KEY_IS_DOWN: action[3] = -1.0 # 视角右转

        # 步进物理环境
        obs, reward, terminated, truncated, info = env.step(action)
        cur_pos = env._getDroneStateVector(0)[0:3]
        
        # 智能追尾相机 (云台减震版)：保持在你身后，方便你看清机头前方的雷达网格
        cam_pos = cam_pos * 0.8 + cur_pos * 0.2
        smooth_yaw = np.degrees(env.target_yaw)
        
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=smooth_yaw - 90, 
            cameraPitch=-20,
            cameraTargetPosition=cam_pos,
            physicsClientId=env.CLIENT
        )

        time.sleep(1/60)
        
        if terminated or truncated:
            if terminated and reward > 100:
                print("命中目标！")
            else:
                print("脱靶或超时！")
            obs, info = env.reset()
            cam_pos = env._getDroneStateVector(0)[0:3].copy()
            
    env.close()

if __name__ == "__main__":
    # manual_control()  
    # train()
    test()

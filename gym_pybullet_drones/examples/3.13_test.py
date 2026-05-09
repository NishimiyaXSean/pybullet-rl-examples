import os
import time
import numpy as np
import pybullet as p
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
        self.EPISODE_LEN_SEC = 10 
        self.pid = DSLPIDControl(drone_model=DroneModel.CF2X)

        self.SMOOTH_FACTOR = 0.1
        self.user_input_pos = np.zeros(3)      
        self.current_target_pos = np.zeros(3)  
        
        # === 核心升级 1：加入 Yaw 控制变量 ===
        self.target_yaw = 0.0 

        # 观察空间依然是 13 维，但意义完全变成了第一人称！
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        
        # === 核心升级 2：动作空间扩增为 4 维 ===
        #[前后速度(vx), 左右速度(vy), 上下速度(vz), 转向速度(yaw_rate)]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs_raw, info = super().reset(seed=seed, options=options)
        
        self.target_pos = np.array([
            np.random.uniform(-5.0, 5.0),
            np.random.uniform(-5.0, 5.0),
            np.random.uniform(0.5, 3.0)
        ])
        
        state = self._getDroneStateVector(0)
        self.prev_dist = np.linalg.norm(self.target_pos - state[0:3])

        self.user_input_pos = state[0:3].copy()
        self.current_target_pos = state[0:3].copy()
        self.target_yaw = 0.0 # 每一局重置视角

        if self.GUI:
            size = 0.2
            p.addUserDebugLine(self.target_pos -[size, 0, 0], self.target_pos +[size, 0, 0],[1, 0, 0], 3)
            p.addUserDebugLine(self.target_pos -[0, size, 0], self.target_pos +[0, size, 0],[0, 1, 0], 3)
            p.addUserDebugLine(self.target_pos -[0, 0, size], self.target_pos +[0, 0, size],[0, 0, 1], 3)
            
            v_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.08, rgbaColor=[1, 0, 0, 0.8])
            self.target_obj_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=v_id, basePosition=self.target_pos)           
            
        return self._computeObs(), info

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        quat = state[3:7] # 无人机的三维空间旋转姿态 (四元数)

        # === 核心升级 3：传感器视觉投影 (World Frame -> Body Frame) ===
        # 1. 计算世界相对距离
        world_rel_pos = self.target_pos - pos
        # 2. 求解无人机当前的逆旋转矩阵
        _, inv_quat = p.invertTransform([0,0,0], quat)
        # 3. 将世界坐标“投影”进无人机的相机坐标系
        local_rel_pos, _ = p.multiplyTransforms([0,0,0], inv_quat, world_rel_pos,[0,0,0,1])
        local_rel_pos = np.array(local_rel_pos) 
        # 现在 local_rel_pos 的含义变成了:[前/后, 左/右, 上/下]

        # 4. 模拟深度相机的硬件噪声 (距离越远，测距越抖)
        dist = np.linalg.norm(local_rel_pos)
        noise = np.random.normal(0, 0.01 + 0.02 * dist, 3) # 基础误差1cm + 2%距离误差
        local_rel_pos += noise

        # 为了让无人机彻底拥有“第一人称”意识，我们把它的速度和虚拟目标点也转成本地坐标
        world_vel = state[10:13]
        local_vel, _ = p.multiplyTransforms([0,0,0], inv_quat, world_vel, [0,0,0,1])
        
        world_virt_pos = self.current_target_pos - pos
        local_virt_pos, _ = p.multiplyTransforms([0,0,0], inv_quat, world_virt_pos,[0,0,0,1])

        rpy = state[7:10]
        z_height = state[2] 
        
        return np.concatenate([local_rel_pos, local_vel, rpy, [z_height], local_virt_pos]).astype(np.float32)

    def step(self, action):
        state = self._getDroneStateVector(0)
        dt = 1 / self.CTRL_FREQ
        quat = state[3:7]

        # === 核心升级 4：第一人称 FPV 动作映射 ===
        # AI 输出的不再是“向世界的东南西北飞”，而是“向我的前后左右飞”
        target_vel_local = action[0:3] * 2.0 
        yaw_rate = action[3] * 1.5 # 允许以 1.5 弧度/秒 的速度转向
        
        self.target_yaw += yaw_rate * dt
        
        # 将第一人称飞行指令转回世界坐标，用于移动我们的“虚拟目标点”
        vel_world, _ = p.multiplyTransforms([0,0,0], quat, target_vel_local, [0,0,0,1])
        vel_world = np.array(vel_world)

        self.user_input_pos += vel_world * dt
        if self.user_input_pos[2] < 0.1:
            self.user_input_pos[2] = 0.1
        
        self.current_target_pos = self.current_target_pos * (1 - self.SMOOTH_FACTOR) + self.user_input_pos * self.SMOOTH_FACTOR
        
        rpm, _, _ = self.pid.computeControl(
            control_timestep=dt,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=self.current_target_pos, 
            target_vel=np.zeros(3),
            target_rpy=np.array([0, 0, self.target_yaw]) # 将偏航角传递给底层 PID 控制器
        )
        
        obs_raw, _, _, truncated, info = super().step(rpm.reshape(1, 4))
        
        new_state = self._getDroneStateVector(0)
        new_dist = np.linalg.norm(self.target_pos - new_state[0:3])
        
        # --- 奖励函数优化 ---
        progress = self.prev_dist - new_dist
        reward = progress * 150.0 
        reward -= 0.05 
        reward -= 0.02 * np.linalg.norm(action)**2 
        
        roll, pitch = new_state[7:9]
        if abs(roll) > 1.0 or abs(pitch) > 1.0:
            reward -= 2.0 

        # === 核心升级 5：视觉锁定奖励 (教它用机头对准红球) ===
        # 在机体坐标系中，正前方是 X 轴 (local_rel[0])
        # 如果 X 轴不是最大正分量，说明无人机没有“看”向红球
        _, inv_quat = p.invertTransform([0,0,0], new_state[3:7])
        local_rel, _ = p.multiplyTransforms([0,0,0], inv_quat, self.target_pos - new_state[0:3],[0,0,0,1])
        local_dist = np.linalg.norm(local_rel)
        if local_dist > 0.2:
            # 计算目标与机头正前方的夹角误差
            angle_error = np.arccos(np.clip(local_rel[0] / local_dist, -1.0, 1.0))
            reward -= 0.1 * angle_error # 视野偏离得越多，扣分越多！

        terminated = False
        if new_dist < 0.08:
            reward += 200 
            terminated = True
            
        if new_state[2] < 0.1 or new_dist > 15.0:
            reward -= 50 
            terminated = True

        self.prev_dist = new_dist
        
        if (self.step_counter / self.PYB_FREQ) > 20.0:
            truncated = True
            
        return self._computeObs(), reward, terminated, truncated, info

# ---------------------------------------------------------
# 训练与测试设置 
# ---------------------------------------------------------

MODEL_PATH = "drone_vision_v1"
VEC_NORM_PATH = "vec_normalize_vision_v1.pkl"

def train():
    env = DummyVecEnv([lambda: DronePIDEnv(gui=False)])
    
    if os.path.exists(VEC_NORM_PATH):
        print("✅ 发现旧的归一化文件，加载中...")
        env = VecNormalize.load(VEC_NORM_PATH, env)
        env.training = True     
        env.norm_reward = True  
    else:
        print("🆕 未找到归一化文件，新建中...")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"✅ 发现旧模型 {MODEL_PATH}.zip，正在加载并进行断点续训！")
        model = PPO.load(MODEL_PATH, env=env) 
    else:
        print("🆕 未找到旧模型，从头开始初始化训练...")
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
        model = PPO("MlpPolicy", env, verbose=1, batch_size=256, 
                    learning_rate=3e-4, n_steps=1024, 
                    policy_kwargs=policy_kwargs, tensorboard_log="./ppo_logs/")

    print("🚀 开始导航训练 (按下 Ctrl+C 可以安全提前停止并保存)...")
    try:
        model.learn(total_timesteps=200_000, reset_num_timesteps=False) 
    except KeyboardInterrupt:
        print("\n🛑 检测到中止信号，正在保存模型...")
    finally:
        model.save(MODEL_PATH)
        env.save(VEC_NORM_PATH) 
        env.close()
        print("💾 模型与环境参数已成功保存！")

def test():
    print("开始演示...")
    env = DummyVecEnv([lambda: DronePIDEnv(gui=True)])
    
    if not os.path.exists(VEC_NORM_PATH):
        print("错误: 找不到 vec_normalize.pkl，请先执行 train()！")
        return
        
    env = VecNormalize.load(VEC_NORM_PATH, env)
    env.training = False 
    env.norm_reward = False

    model = PPO.load(MODEL_PATH)
    base_env = env.venv.envs[0]
    
    obs = env.reset()
    prev_pos = base_env._getDroneStateVector(0)[0:3]

    # === 🎥 虚拟云台初始化 ===
    cam_pos = prev_pos.copy() 

    episode_count = 0

    while episode_count < 20: 
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        cur_pos = base_env._getDroneStateVector(0)[0:3]
        
        if not done[0]:
            p.addUserDebugLine(prev_pos, cur_pos, [1, 0, 0], 2.5, 1.5, physicsClientId=base_env.CLIENT)
            prev_pos = cur_pos
            
            # 让摄像机像被橡皮筋拉着一样跟随机体，吸收高频震动 (0.9保留历史，0.1追踪当前)
            cam_pos = cam_pos * 0.9 + cur_pos * 0.1
            # 不再读取物理的 rpy[2]，而是直接读取环境里完美的数学 target_yaw
            smooth_yaw = np.degrees(base_env.target_yaw)

            p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw= smooth_yaw - 90, 
                cameraPitch=-20,
                cameraTargetPosition=cam_pos
            )
            
        time.sleep(1/30) 
        
        if done[0]:
            episode_count += 1
            print(f"第 {episode_count} 轮测试结束！")
            prev_pos = base_env._getDroneStateVector(0)[0:3]
            cam_pos = prev_pos.copy() # 重置云台位置
            
    env.close()

def manual_control():
    print("==================================")
    print("🎮 开启第一人称 (FPV) 手动驾驶模式！")
    print("操作说明 (请确保鼠标点击激活了 PyBullet 仿真窗口)：")
    print("  [W] / [S] : 前进 / 后退 (本地 X 轴)")
    print("  [A] / [D] : 转向左 / 转向右 (Yaw 偏航控制!)")
    print("  [↑] / [↓] : 升高 / 降低 (Z轴)")
    print("[←] / [→] : 侧飞左 / 侧飞右 (本地 Y 轴)")
    print("  [R]       : 重置环境 | [Q] 退出")
    print("==================================")

    env = DronePIDEnv(gui=True)
    obs, info = env.reset()
    prev_pos = env._getDroneStateVector(0)[0:3]

    # === 🎥 虚拟云台初始化 ===
    cam_pos = prev_pos.copy()

    while True:
        action = np.zeros(4, dtype=np.float32) # 现在是 4 维动作
        keys = p.getKeyboardEvents()
        
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED: break
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            obs, info = env.reset()
            prev_pos = env._getDroneStateVector(0)[0:3]
            cam_pos = prev_pos.copy() # 重置云台
            continue

        # 第一人称操作映射
        if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN: action[0] = 1.0
        if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN: action[0] = -1.0
        
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN: action[1] = 1.0
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN: action[1] = -1.0
            
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN: action[2] = 1.0
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN: action[2] = -1.0
            
        if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN: action[3] = 1.0 # 视角左转
        if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN: action[3] = -1.0 # 视角右转

        obs, reward, terminated, truncated, info = env.step(action)
        cur_pos = env._getDroneStateVector(0)[0:3]
        
        p.addUserDebugLine(prev_pos, cur_pos, [0, 1, 1], 2.5, 1.5, physicsClientId=env.CLIENT)
        prev_pos = cur_pos

        carrot = env.current_target_pos
        p.addUserDebugLine(carrot-[0.1,0,0], carrot+[0.1,0,0],[1,0,1], 2, 0.05, physicsClientId=env.CLIENT)
        p.addUserDebugLine(carrot-[0,0.1,0], carrot+[0,0.1,0],[1,0,1], 2, 0.05, physicsClientId=env.CLIENT)

        # 智能追尾相机——手动模式下，为了让操作手感更紧凑，跟得稍微紧一点 (0.8保留，0.2追踪)
        cam_pos = cam_pos * 0.8 + cur_pos * 0.2
        smooth_yaw = np.degrees(env.target_yaw)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=smooth_yaw - 90, 
            cameraPitch=-20,
            cameraTargetPosition=cam_pos
        )

        time.sleep(1/30)
        if terminated or truncated:
            obs, info = env.reset()
            prev_pos = env._getDroneStateVector(0)[0:3]
            cam_pos = prev_pos.copy()
            
    env.close()

if __name__ == "__main__":
    # manual_control()
    # train()
    test()
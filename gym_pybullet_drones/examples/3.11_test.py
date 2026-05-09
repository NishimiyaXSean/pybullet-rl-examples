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
            gui=gui
        )
        self.pid = DSLPIDControl(drone_model=DroneModel.CF2X)
        
        # 优化1：扩充观测空间为 9 维 (相对位置[3] + 速度[3] + 姿态角RPY[3])
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        # 动作空间：[vx, vy, vz] 保持不变
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs_raw, info = super().reset(seed=seed, options=options)
        
        # 缩小目标范围
        self.target_pos = np.array([
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(0.5, 1.5)
        ])
        
        state = self._getDroneStateVector(0)
        self.prev_dist = np.linalg.norm(self.target_pos - state[0:3])
        
        if self.GUI:
            p.removeAllUserDebugItems()
            
            # 十字架辅助线
            size = 0.2
            p.addUserDebugLine(self.target_pos - [size, 0, 0], self.target_pos +[size, 0, 0], [1, 0, 0], 3)
            p.addUserDebugLine(self.target_pos - [0, size, 0], self.target_pos +[0, size, 0], [0, 1, 0], 3)
            p.addUserDebugLine(self.target_pos - [0, 0, size], self.target_pos +[0, 0, size], [0, 0, 1], 3)
            
            # 优化2：防止内存泄漏，只创建一次球体，后续只移动它
            if self.target_obj_id == -1:
                v_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.08, rgbaColor=[1, 0, 0, 0.8])
                self.target_obj_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=v_id, basePosition=self.target_pos)
            else:
                p.resetBasePositionAndOrientation(self.target_obj_id, self.target_pos,[0, 0, 0, 1])
            
            p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=-30, cameraPitch=-30, cameraTargetPosition=[0, 0, 1])
        
        return self._computeObs(), info

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        rel_pos = self.target_pos - state[0:3]
        rpy = state[7:10] # 获取 Roll, Pitch, Yaw
        vel = state[10:13]
        
        # 将信息拼接 (稍后会在 VecNormalize 中自动归一化)
        return np.concatenate([rel_pos, vel, rpy]).astype(np.float32)

    def step(self, action):
        state = self._getDroneStateVector(0)
        
        # 映射动作到目标速度 (最大 1.5 m/s)
        target_vel = action * 1.5 
        
        # 优化3：修复PID控制逻辑 (使用胡萝卜原理：目标位置 = 当前位置 + 目标速度 * 时间步)
        dt = 1 / self.CTRL_FREQ
        target_pos_carrot = state[0:3] + target_vel * dt
        
        rpm, _, _ = self.pid.computeControl(
            control_timestep=dt,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=target_pos_carrot, # 关键修改：传入动态的引导点
            target_vel=target_vel
        )
        
        obs_raw, _, _, truncated, info = super().step(rpm.reshape(1, 4))
        
        # --- 奖励函数优化 ---
        new_state = self._getDroneStateVector(0)
        new_dist = np.linalg.norm(self.target_pos - new_state[0:3])
        
        # 1. 进度奖励 (接近给正奖，远离给负奖)
        progress = self.prev_dist - new_dist
        reward = progress * 150.0 
        
        # 2. 距离惩罚 (鼓励靠近)
        reward -= new_dist * 0.05
        
        # 3. 动作平滑惩罚 (防止乱飞和高频震荡)
        reward -= 0.02 * np.linalg.norm(action)
        
        terminated = False
        
        # 4. 成功大奖
        if new_dist < 0.2:
            reward += 100
            terminated = True
        
        # 5. 危险姿态与坠毁惩罚
        roll, pitch = new_state[7:9]
        if abs(roll) > 1.2 or abs(pitch) > 1.2: # 姿态过于倾斜
            reward -= 10
            
        if new_state[2] < 0.05 or new_dist > 3.0: # 坠毁或飞离太远
            reward -= 50 # 不要设置太大，否则AI会为了避免负分而在开始就故意坠毁结束
            terminated = True

        self.prev_dist = new_dist
        
        return self._computeObs(), reward, terminated, truncated, info

# ---------------------------------------------------------
# 训练与测试设置
# ---------------------------------------------------------

def train():
    # 使用 DummyVecEnv 包装，方便后续扩展为多线程
    env = DummyVecEnv([lambda: DronePIDEnv(gui=False)])
    
    # 优化4：使用 VecNormalize 自动对观测状态和奖励进行归一化，大幅提升 PPO 收敛速度
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 优化5：调整超参数 (减小学习率，增加网络深度)
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    model = PPO("MlpPolicy", 
                env, 
                verbose=1, 
                batch_size=256,       # 增加 Batch
                learning_rate=3e-4,   # PPO 标准稳妥学习率
                n_steps=1024,
                policy_kwargs=policy_kwargs,
                tensorboard_log="./ppo_logs/")

    print("开始导航训练...")
    # 增加训练步数
    model.learn(total_timesteps=500_000) 
    model.save("drone_pid_v4")
    
    # 必须保存 Normalizer 的状态，否则测试时输入的数据分布就不对了！
    env.save("vec_normalize.pkl") 
    env.close()

def test():
    print("开始演示...")
    # 测试时也要用完全相同的包装方式
    env = DummyVecEnv([lambda: DronePIDEnv(gui=True)])
    env = VecNormalize.load("vec_normalize.pkl", env)
    
    # 测试时不需要更新归一化参数
    env.training = False 
    env.norm_reward = False

    model = PPO.load("drone_pid_v4")
    
    for _ in range(10):
        obs = env.reset() # 注意 VecEnv 的 reset 不返回 info
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            # DummyVecEnv 把 terminated 和 truncated 合并为了 done
            time.sleep(1/30) # 减慢演示速度
    env.close()

if __name__ == "__main__":
    # 第一次运行请取消注释 train()
    # train()
    test()
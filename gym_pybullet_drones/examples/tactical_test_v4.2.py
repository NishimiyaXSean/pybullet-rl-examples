import os
import time
import numpy as np
import pybullet as p
import gymnasium as gym
from stable_baselines3 import PPO

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class DronePIDEnv(CtrlAviary):
    def __init__(self, gui=False):
        self.target_pos = np.array([1.0, 1.0, 1.0])
        self.target_obj_id = -1
        
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
        # 观察：相对位置(3) + 速度(3) + 姿态RPY(3) = 9维
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs_raw, info = super().reset(seed=seed, options=options)
        # 目标点范围缩小到可见范围内
        self.target_pos = np.array([
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(0.8, 1.4)
        ])
        
        if self.GUI:
            p.removeAllUserDebugItems()
            # 画一个明显的十字架引导
            s = 0.15
            p.addUserDebugLine(self.target_pos-[s,0,0], self.target_pos+[s,0,0], [1,0,0], 3)
            p.addUserDebugLine(self.target_pos-[0,s,0], self.target_pos+[0,s,0], [0,1,0], 3)
            p.addUserDebugLine(self.target_pos-[0,0,s], self.target_pos+[0,0,s], [0,0,1], 3)
            v_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.08, rgbaColor=[1, 0, 0, 0.8])
            self.target_obj_id = p.createMultiBody(0, v_id, basePosition=self.target_pos)

        return self._computeObs(), info

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        # 相对位置进行归一化处理 (除以3)
        rel_pos = (self.target_pos - state[0:3]) / 3.0
        vel = state[10:13] / 2.0
        rpy = state[7:10] # 姿态本身就在-pi到pi
        return np.concatenate([rel_pos, vel, rpy]).astype(np.float32).flatten()

    def step(self, action):
        state = self._getDroneStateVector(0)
        # 将RL动作映射为期望速度向量
        target_vel = action * 1.0 # 最大速度限制在1m/s，更稳
        
        rpm, _, _ = self.pid.computeControl(
            control_timestep=1/self.CTRL_FREQ,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=state[0:3], 
            target_vel=target_vel
        )
        
        obs_raw, _, _, truncated, info = super().step(rpm.reshape(1, 4))
        
        # --- 强引导奖励函数 ---
        new_state = self._getDroneStateVector(0)
        pos = new_state[0:3]
        vel = new_state[10:13]
        target_vec = self.target_pos - pos
        dist = np.linalg.norm(target_vec)
        
        # 1. 基础距离奖励 (使用指数函数，越近奖励增长越快)
        reward = 1.0 / (1.0 + dist)
        
        # 2. 【核心】方向引导奖励
        # 计算速度向量和目标向量的夹角余弦值
        if dist > 0.1 and np.linalg.norm(vel) > 0.05:
            unit_vel = vel / np.linalg.norm(vel)
            unit_target = target_vec / dist
            direction_match = np.dot(unit_vel, unit_target) # -1 到 1
            reward += direction_match * 0.5 # 如果朝着球飞，大幅加分
        
        # 3. 姿态惩罚 (鼓励水平飞行)
        reward -= (abs(new_state[7]) + abs(new_state[8])) * 0.1

        terminated = False
        if dist < 0.15: # 成功
            reward += 100
            terminated = True
        
        if pos[2] < 0.1 or dist > 4.0: # 失败
            reward -= 20
            terminated = True

        return self._computeObs(), reward, terminated, truncated, info

# ---------------------------------------------------------

def run_fast():
    # 1. 训练
    train_env = DronePIDEnv(gui=False)
    # 使用比较激进的学习率，快速建立联系
    model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=1e-3)
    
    print("开始强化训练：重点关注方向引导...")
    model.learn(total_timesteps=100000)
    model.save("drone_strong_guide")
    train_env.close()

    # 2. 测试
    test_env = DronePIDEnv(gui=True)
    model = PPO.load("drone_strong_guide")
    
    for _ in range(10):
        obs, info = test_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-30, 
                                         cameraPitch=-30, cameraTargetPosition=obs[0:3]*3.0) # 还原真实位置
            time.sleep(1/30)
    test_env.close()

if __name__ == "__main__":
    run_fast()
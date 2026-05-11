import os
import time
import numpy as np
import pybullet as p
import pybullet_data
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
        self.prev_dist = 0.0 # 用于计算进度奖励
        
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=np.array([[0, 0, 1.0]]), # 空中起步
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=30,
            gui=gui
        )
        self.pid = DSLPIDControl(drone_model=DroneModel.CF2X)
        # 观察：相对位置(3) + 速度(3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        # 动作：[vx, vy, vz]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs_raw, info = super().reset(seed=seed, options=options)
        
        # 缩小目标范围，让它更容易学到 [±1.0, ±1.0, 0.8~1.5]
        self.target_pos = np.array([
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(0.8, 1.5)
        ])
        
        # 初始化距离记录
        state = self._getDroneStateVector(0)
        self.prev_dist = np.linalg.norm(self.target_pos - state[0:3])
        
        if self.GUI:
            # 清除之前的辅助线
            p.removeAllUserDebugItems()
            
            # 画一个持久的 3D 十字架，确保你能看到目标位置
            size = 0.2
            p.addUserDebugLine(self.target_pos - [size, 0, 0], self.target_pos + [size, 0, 0], [1, 0, 0], 3)
            p.addUserDebugLine(self.target_pos - [0, size, 0], self.target_pos + [0, size, 0], [0, 1, 0], 3)
            p.addUserDebugLine(self.target_pos - [0, 0, size], self.target_pos + [0, 0, size], [0, 0, 1], 3)
            
            # 创建红球
            v_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.08, rgbaColor=[1, 0, 0, 0.8])
            self.target_obj_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=v_id, basePosition=self.target_pos)
            
            # 摄像头初始对准目标
            p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=-30, cameraPitch=-30, cameraTargetPosition=[0, 0, 1])
        
        return self._computeObs(), info

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        rel_pos = self.target_pos - state[0:3]
        vel = state[10:13]
        return np.concatenate([rel_pos, vel]).astype(np.float32).flatten()

    def step(self, action):
        state = self._getDroneStateVector(0)
        
        # 映射 RL 动作到期望速度
        target_vel = action * 1.5 # 最大速度 1.5m/s
        
        # 底层 PID 执行
        rpm, _, _ = self.pid.computeControl(
            control_timestep=1/self.CTRL_FREQ,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=state[0:3], # 控速度模式
            target_vel=target_vel
        )
        
        obs_raw, _, _, truncated, info = super().step(rpm.reshape(1, 4))
        
        # --- 核心：奖励函数重构 ---
        new_state = self._getDroneStateVector(0)
        new_dist = np.linalg.norm(self.target_pos - new_state[0:3])
        
        # 1. 进度奖励：每靠近一厘米都有奖金 (由于一帧移动很短，放大倍数)
        progress = self.prev_dist - new_dist
        reward = progress * 100.0 
        
        # 2. 距离惩罚：基础生存压力
        reward -= new_dist * 0.1
        
        # 3. 时间惩罚：鼓励尽快到达，不许原地挂机
        reward -= 0.1
        
        # 4. 成功大奖
        terminated = False
        if new_dist < 0.2:
            reward += 200
            terminated = True
        
        # 5. 失败惩罚
        if new_state[2] < 0.1 or abs(new_state[7]) > 1.0:
            reward -= 50

        if new_state[2] < 0.05 or new_dist > 4.0: # 坠毁或飞丢
            reward -= 100
            terminated = True

        self.prev_dist = new_dist # 更新距离记录
        
        return self._computeObs(), reward, terminated, truncated, info

# ---------------------------------------------------------
# 训练设置 (强化版)
# ---------------------------------------------------------

def train():
    # 1. 训练
    train_env = DronePIDEnv(gui=False)
    
    # 调大 batch_size 让学习更稳定
    # 调小 learning_rate 让学习更细腻
    model = PPO("MlpPolicy", 
                train_env, 
                verbose=1, 
                batch_size=128,
                learning_rate=1e-3, 
                tensorboard_log="./ppo_logs/")

    print("开始导航训练 (Progress Reward 模式)...")
    # 如果 ep_rew_mean 持续上涨，说明方向对了
    model.learn(total_timesteps=120000)
    model.save("drone_pid_v3")
    train_env.close()


def test():
    # 2. 演示
    print("开始演示...")
    test_env = DronePIDEnv(gui=True)
    model = PPO.load("drone_pid_v3")
    
    for _ in range(10):
        obs, info = test_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated

            # 实时更新摄像头焦点在无人机上
            # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-30, 
            #                              cameraPitch=-30, cameraTargetPosition=obs[0:3])
            # time.sleep(1/30)
    test_env.close()


if __name__ == "__main__":
    # train()
    test()
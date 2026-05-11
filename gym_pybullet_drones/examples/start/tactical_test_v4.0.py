import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import time
import numpy as np
import pybullet as p
import gymnasium as gym
from stable_baselines3 import PPO
# from stable_baselines3.common.env_checker import check_env

# 导入 gym-pybullet-drones 的基础环境
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class DronePIDEnv(CtrlAviary):
    """ 自定义强化学习环境：飞向随机目标点 """
    
    def __init__(self, gui=False):
        # 初始化基础物理环境
        def __init__(self, gui=False):
            self.target_pos = np.array([1.0, 1.0, 1.0])
            self.target_obj_id = -1
        
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=np.array([[0, 0, 1.0]]), 
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=30, # RL 决策频率
            gui=gui
        )
        
        # 核心：引入一个底层 PID 控制器
        self.pid = DSLPIDControl(drone_model=DroneModel.CF2X)
        
        # 定义目标点范围：在以原点为中心的 2x2x2 空间内
        self.target_pos = np.array([1.0, 1.0, 1.0])
        self.target_obj_id = -1 # 初始化为一个无效 ID

        
        # 观察空间：相对位置(3) + 当前速度(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(6,), 
            dtype=np.float32
        )
        # 动作空间：期望的 3D 速度向量 [vx, vy, vz]，范围 -1 到 1 (m/s)
        self.action_space = gym.spaces.Box(
            low=-1, 
            high=1, 
            shape=(3,), 
            dtype=np.float32
        )

        # 设置动作空间：[vx, vy, vz, yaw_speed] 范围在 -1 到 1 之间
        # 我们后续会将其映射到实际的速度值
        def _actionSpace(self):
            return gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        # 设置观察空间
        def _observationSpace(self):
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """ 重置环境：随机生成一个新的目标点 """

        # 处理随机种子
        obs_base, info = super().reset(seed=seed, options=options)

        # 随机目标点
        self.target_pos = np.array([
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(0.5, 1.5)
        ])
        
        # 在 GUI 中画出目标点位置（小红球）
        if self.GUI:
            # 如果 ID 有效，先尝试从场景删除旧球，防止重置导致 ID 失效
            try:
                p.removeBody(self.target_obj_id)
            except:
                pass 
            
            # 重新创建红球
            v_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1])
            self.target_obj_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=v_id, basePosition=self.target_pos)
            
        return self._computeObs(), info

    def _computeObs(self):
        """ 构建观察向量 """
        state = self._getDroneStateVector(0) # 获取第0架飞机的状态
        pos = state[0:3]
        vel = state[10:13]
        # rpy = state[7:10]
        
        # 相对坐标
        rel_pos = self.target_pos - pos
        
        obs = np.concatenate([rel_pos, vel]).astype(np.float32)
        return obs

    def _computeReward(self):
        """ 设计奖励函数"""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        
        # 1. 距离惩罚（离得越远，扣分越多）
        dist = np.linalg.norm(self.target_pos - pos)
        reward = -dist
        
        # 2. 到达奖
        if dist < 0.15:
            reward += 100
            
        # 3. 翻车惩罚
        if abs(state[7]) > 1.2 or abs(state[8]) > 1.2:
            reward -= 10
            
        return reward

    def _computeTerminated(self):
        """ 判断回合是否提前结束 """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        dist = np.linalg.norm(self.target_pos - pos)
        
        # 成功到达或彻底坠毁
        if dist < 0.2 or pos[2] < 0.01 or abs(state[7]) > 2.0 or abs(state[8]) > 2.0:
            return True
        return False

    def _computeTruncated(self):
        """ 超时判断 """
        if self.step_counter / self.CTRL_FREQ > 15: # 15秒没到就重来
            return True
        return False

    def step(self, action):
        """ 
        RL 输出动作，PID 转换为转速。
        action: [vx, vy, vz]
        """
        state = self._getDroneStateVector(0)
        
        # 映射 RL 动作到期望速度 (最大 1.5 m/s)
        target_vel = action * 1.5
        
        # 使用 PID 计算达成该速度所需的电机 RPM
        rpm, _, _ = self.pid.computeControl(
            control_timestep=1/self.CTRL_FREQ,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=state[0:3], # 我们控速度，所以 target_pos 设为当前位置即可
            target_vel=target_vel
        )
        # 调用父类的底层 step (它期待 RPM 作为输入)
        # 注意：CtrlAviary 的动作形状通常是 (1, 4)
        obs_raw, reward, terminated, truncated, info = super().step(rpm.reshape(1, 4))
        
        # 计算我们自定义的奖励和观察
        return self._computeObs(), self._computeReward(), self._computeTerminated(), truncated, info

    def _computeInfo(self):
        return {} # 暂不需要额外信息

# ==========================================
# 训练与测试脚本
# ==========================================

def train():
    # 1. 创建训练环境 (不带 GUI 速度快)
    env = DronePIDEnv(gui=False)

    # 2. 检查环境是否符合规范
    # check_env(train_env) 

    # 3. 定义 PPO 模型
    # MlpPolicy 表示全连接神经网络
    model = PPO("MlpPolicy", 
                env, 
                verbose=1, 
                learning_rate=1e-3, 
                tensorboard_log="./ppo_drone_logs/")

    # 4. 开始训练
    print("正在开始训练（阶段1：前往随机目标点）..")
    model.learn(total_timesteps=80000)
    
    # 5. 保存模型
    model.save("drone_pid_rl_v1")
    print("模型已保存。")
    env.close()

def test():
    # 1. 加载模型
    print("加载模型并开始演示...")
    model = PPO.load("drone_pid_rl_v1")
    
    # 2. 创建测试环境 (带 GUI 观看)
    test_env = DronePIDEnv(gui=True)
    obs, _ = test_env.reset()
    
    for _ in range(10): # 测试10个回合
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            time.sleep(1/30) # 减慢速度方便观察
        obs, _ = test_env.reset()
    
    test_env.close()

if __name__ == "__main__":
    # train()
    test()
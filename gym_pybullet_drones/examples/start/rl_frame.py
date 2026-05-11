import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO # 推荐算法：PPO
from stable_baselines3.common.env_checker import check_env
from gym_pybullet_drones.envs.HoverAviary import HoverAviary # 基础环境

# 1. 为什么不直接用自带环境？因为我们要“自定义随机障碍物和目标”
class ObstacleAvoidanceEnv(HoverAviary):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 在这里定义你的目标点
        self.target_pos = np.array([2.0, 2.0, 1.0])
        
    def _computeObs(self):
        # 【核心】这里返回无人机看到的所有信息
        # 例如：[当前位置, 速度, 到目标的距离, 射线距离]
        obs = super()._computeObs() 
        return obs # 实际上你需要在这里拼入传感器数据

    def _computeReward(self):
        # 【核心】设计奖励
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(state[0:3] - self.target_pos)
        
        reward = -dist # 离目标越近，分越高
        if dist < 0.1: reward += 100 # 到达奖
        return reward

    def _computeTerminated(self):
        # 【核心】判断是否撞墙或翻车
        state = self._getDroneStateVector(0)
        if state[2] < 0.1 or abs(state[7]) > 1.0: # 撞地或翻车
            return True
        return False

# 2. 训练脚本
if __name__ == "__main__":
    # 创建训练环境
    env = ObstacleAvoidanceEnv(gui=False) # 训练时关闭 GUI 速度快几十倍
    
    # 初始化算法
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_drone_tensorboard/")
    
    # 开始“试错”学习
    print("开始训练...")
    model.learn(total_timesteps=100000) # 建议起步 10万步
    
    # 保存模型
    model.save("drone_avoidance_model")
    
    # 3. 演示训练结果
    env = ObstacleAvoidanceEnv(gui=True)
    obs, _ = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated: obs, _ = env.reset()
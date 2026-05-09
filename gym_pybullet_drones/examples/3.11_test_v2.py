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
        
        # 观测空间: 相对位置(3) + 速度(3) + 姿态RPY(3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        # 动作空间: [vx, vy, vz]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs_raw, info = super().reset(seed=seed, options=options)
        
        # 训练时可以把范围稍微拉大，测试其远距离飞行能力
        self.target_pos = np.array([
            np.random.uniform(-1.5, 1.5),
            np.random.uniform(-1.5, 1.5),
            np.random.uniform(0.5, 1.5)
        ])
        
        state = self._getDroneStateVector(0)
        self.prev_dist = np.linalg.norm(self.target_pos - state[0:3])
        
        if self.GUI:
            # 修复1：因为 super().reset() 会清空物理世界，必须每次重新创建红球
            size = 0.2
            p.addUserDebugLine(self.target_pos - [size, 0, 0], self.target_pos +[size, 0, 0], [1, 0, 0], 3)
            p.addUserDebugLine(self.target_pos -[0, size, 0], self.target_pos +[0, size, 0], [0, 1, 0], 3)
            p.addUserDebugLine(self.target_pos -[0, 0, size], self.target_pos +[0, 0, size], [0, 0, 1], 3)
            
            v_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.08, rgbaColor=[1, 0, 0, 0.8])
            self.target_obj_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=v_id, basePosition=self.target_pos)
            
            p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=-45, cameraPitch=-30, cameraTargetPosition=[0, 0, 1])
        
        return self._computeObs(), info

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        rel_pos = self.target_pos - state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        return np.concatenate([rel_pos, vel, rpy]).astype(np.float32)

    def step(self, action):
        state = self._getDroneStateVector(0)
        
        # 修复3：提升最大速度，并拉大“引导胡萝卜”的距离，让底层PID产生更大的姿态倾斜
        target_vel = action * 2.0 
        target_pos_carrot = state[0:3] + target_vel * 0.1 # 乘数从 dt 增大到 0.1，引导点更靠前
        
        rpm, _, _ = self.pid.computeControl(
            control_timestep=1/self.CTRL_FREQ,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=target_pos_carrot, 
            target_vel=target_vel
        )
        
        obs_raw, _, _, truncated, info = super().step(rpm.reshape(1, 4))
        
        new_state = self._getDroneStateVector(0)
        new_dist = np.linalg.norm(self.target_pos - new_state[0:3])
        
        # --- 针对远距离“自杀坠机”优化的奖励函数 ---
        
        # 1. 进度奖励（纯粹基于势能变化，靠近多少给多少，非常公平）
        progress = self.prev_dist - new_dist
        reward = progress * 150.0 
        
        # 2. 存活时间惩罚（只给微小的时间惩罚，逼迫它不要悬停）
        reward -= 0.05
        
        # 3. 动作平滑惩罚（防止发癫）
        reward -= 0.02 * np.linalg.norm(action)**2
        
        terminated = False
        
        # 4. 成功大奖（到达立刻结束）
        if new_dist < 0.2:
            reward += 200 # 奖励足够大，它才会觉得长途跋涉是值得的
            terminated = True
            
        # 5. 坠毁或飞丢惩罚
        if new_state[2] < 0.05 or new_dist > 4.0:
            reward -= 50 # 惩罚不能比远距离飞行的总扣分还低，否则它会选择自杀
            terminated = True

        self.prev_dist = new_dist
        
        return self._computeObs(), reward, terminated, truncated, info

# ---------------------------------------------------------
# 训练与测试设置
# ---------------------------------------------------------

def train():
    env = DummyVecEnv([lambda: DronePIDEnv(gui=False)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    model = PPO("MlpPolicy", 
                env, 
                verbose=1, 
                batch_size=256,
                learning_rate=3e-4, 
                n_steps=1024,
                policy_kwargs=policy_kwargs,
                tensorboard_log="./ppo_logs/")

    print("开始导航训练 (按下 Ctrl+C 可以安全提前停止并保存)...")
    try:
        model.learn(total_timesteps=600_000) 
    except KeyboardInterrupt:
        print("\n检测到中止信号，正在保存模型...")
    finally:
        model.save("drone_pid_v5")
        env.save("vec_normalize.pkl") 
        env.close()
        print("保存成功！")

def test():
    print("开始演示...")
    env = DummyVecEnv([lambda: DronePIDEnv(gui=True)])
    
    # 确保之前训练时生成了 vec_normalize.pkl
    if not os.path.exists("vec_normalize.pkl"):
        print("错误: 找不到 vec_normalize.pkl，请先执行 train()！")
        return
        
    env = VecNormalize.load("vec_normalize.pkl", env)
    env.training = False 
    env.norm_reward = False

    model = PPO.load("drone_pid_v5")
    
    # 修复2：正确处理 DummyVecEnv 的测试循环
    for i in range(10): # 测试 10 次
        obs = env.reset()
        episode_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            # DummyVecEnv 返回的 done 是一个由 True/False 组成的数组
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            
            # 放慢点时间，不然你看不到它飞
            time.sleep(1/30) 
            
            # 如果这一个 episode 结束了 (到达终点或坠毁)
            if done[0]:
                print(f"第 {i+1} 轮测试结束！")
                break # 立刻跳出当前循环，进入下一个 reset

    env.close()

if __name__ == "__main__":
    # train()
    test()
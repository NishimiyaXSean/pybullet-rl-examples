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
        # 修改最大回合时间至 10 秒
        self.EPISODE_LEN_SEC = 10 
        self.pid = DSLPIDControl(drone_model=DroneModel.CF2X)
        
        # 核心修复1：观测空间增至 10 维 (相对位置[3] + 速度[3] + 姿态RPY[3] + 绝对高度[1])
        # 加上绝对高度，无人机就能学到“快撞地时不能猛低头”
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs_raw, info = super().reset(seed=seed, options=options)
        
        # 随机生成目标，包含远距离和近距离，形成自然课程学习
        self.target_pos = np.array([
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(0.5, 2.0)
        ])
        
        state = self._getDroneStateVector(0)
        self.prev_dist = np.linalg.norm(self.target_pos - state[0:3])
        
        if self.GUI:
            size = 0.2
            p.addUserDebugLine(self.target_pos - [size, 0, 0], self.target_pos +[size, 0, 0], [1, 0, 0], 3)
            p.addUserDebugLine(self.target_pos -[0, size, 0], self.target_pos +[0, size, 0], [0, 1, 0], 3)
            p.addUserDebugLine(self.target_pos -[0, 0, size], self.target_pos +[0, 0, size], [0, 0, 1], 3)
            
            v_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.08, rgbaColor=[1, 0, 0, 0.8])
            self.target_obj_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=v_id, basePosition=self.target_pos)
            
            p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=-45, cameraPitch=-30, cameraTargetPosition=[0, 0, 1])
        
        return self._computeObs(), info

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        rel_pos = self.target_pos - state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        z_height = state[2] # 提取绝对高度
        # 将绝对高度拼接进去
        return np.concatenate([rel_pos, vel, rpy, [z_height]]).astype(np.float32)

    def step(self, action):
        state = self._getDroneStateVector(0)
        
        # 核心修复2：将引导时间步回调为真实的 dt。
        # 放弃激进的超前引导，宁愿飞得慢一点，也要保证姿态平稳不掉高度
        dt = 1 / self.CTRL_FREQ
        target_vel = action * 2.0 
        target_pos_carrot = state[0:3] + target_vel * dt 
        
        rpm, _, _ = self.pid.computeControl(
            control_timestep=dt,
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
        
        # --- 奖励函数 ---
        progress = self.prev_dist - new_dist
        reward = progress * 150.0 
        
        reward -= 0.05 # 微小存活惩罚
        reward -= 0.02 * np.linalg.norm(action)**2 # 动作平滑惩罚
        
        # 对危险姿态施加额外惩罚，教导它不要为了加速而玩命低头
        roll, pitch = new_state[7:9]
        if abs(roll) > 1.0 or abs(pitch) > 1.0:
            reward -= 2.0 

        terminated = False
        
        if new_dist < 0.2:
            reward += 200 
            terminated = True
            
        if new_state[2] < 0.1 or new_dist > 5.0:
            reward -= 50 
            terminated = True

        self.prev_dist = new_dist
        return self._computeObs(), reward, terminated, truncated, info

# ---------------------------------------------------------
# 训练与测试设置 (支持断点续训)
# ---------------------------------------------------------

MODEL_PATH = "drone_pid_v5"
VEC_NORM_PATH = "vec_normalize#3.12_test.pkl"

def train():
    env = DummyVecEnv([lambda: DronePIDEnv(gui=False)])
    
    # 1. 尝试加载归一化参数
    if os.path.exists(VEC_NORM_PATH):
        print("✅ 发现旧的归一化文件，加载中...")
        env = VecNormalize.load(VEC_NORM_PATH, env)
        env.training = True     # 必须开启，继续更新数据分布
        env.norm_reward = True  # 必须开启，继续归一化奖励
    else:
        print("🆕 未找到归一化文件，新建中...")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 2. 尝试加载旧模型
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"✅ 发现旧模型 {MODEL_PATH}.zip，正在加载并进行断点续训！")
        # 关键：加载时必须把 env 传进去绑定
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
    
    # 通过 .venv.envs[0] 剥开 VecNormalize 和 DummyVecEnv 的包装，获取底层的 pybullet 环境实例
    base_env = env.venv.envs[0]
    
    for i in range(20): 
        obs = env.reset()

        # 在每一轮开始时，获取无人机的初始绝对坐标 (X, Y, Z)
        prev_pos = base_env._getDroneStateVector(0)[0:3]

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # 获取执行动作后，无人机当前的新坐标
            cur_pos = base_env._getDroneStateVector(0)[0:3]
            
            # 调用 PyBullet 画一条从旧坐标到新坐标的绿色线段
            p.addUserDebugLine(
                lineFromXYZ=prev_pos, 
                lineToXYZ=cur_pos, 
                lineColorRGB=[1, 0, 0],   # [R, G, B] 颜色，这里是纯绿色
                lineWidth=2.5,            # 线条粗细
                lifeTime=1.5,               # 0 代表永久保留，直到这局结束被 reset 清空
                physicsClientId=base_env.CLIENT # 指定在哪个 PyBullet 窗口画图
            )
            
            # 更新坐标，为下一帧画线做准备
            prev_pos = cur_pos
            
            time.sleep(1/30) 
            
            if done[0]:
                print(f"第 {i+1} 轮测试结束！")
                break 

    env.close()

def manual_control():
    print("==================================")
    print("🛸 开启手动控制模式！")
    print("操作说明 (请确保鼠标点击激活了 PyBullet 仿真窗口)：")
    print("  [W] / [S] : 前进 / 后退 (X轴)")
    print("  [A] / [D] : 向左 / 向右 (Y轴)")
    print("  [↑] / [↓] : 上升 / 下降 (Z轴)")
    print("  [R]       : 手动重置环境")
    print("[Q]       : 退出手动模式")
    print("==================================")

    # 手动模式下不需要使用 DummyVecEnv 包装，直接实例化底层环境即可
    env = DronePIDEnv(gui=True)
    obs, info = env.reset()
    
    # 获取初始位置用于画拖尾
    prev_pos = env._getDroneStateVector(0)[0:3]

    while True:
        # 初始化动作为全 0 (代表悬停)
        action = np.zeros(3, dtype=np.float32)
        
        # 获取 PyBullet 的键盘事件
        keys = p.getKeyboardEvents()
        
        # 1. 退出与重置指令
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            print("退出手动模式...")
            break
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            print("手动重置环境！")
            obs, info = env.reset()
            prev_pos = env._getDroneStateVector(0)[0:3]
            continue

        # 2. X轴控制 (前后) - 键盘 W 和 S
        if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
            action[0] = 1.0  # 全速前进
        if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
            action[0] = -1.0 # 全速后退
            
        # 3. Y轴控制 (左右) - 键盘 A 和 D
        if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
            action[1] = 1.0  # 全速向左
        if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
            action[1] = -1.0 # 全速向右
            
        # 4. Z轴控制 (上下) - 键盘 上下箭头
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            action[2] = 1.0  # 全速上升
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            action[2] = -1.0 # 全速下降

        # 将按键转换为动作输入给环境
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 画出你的飞行拖尾 (保留之前我们加的流星拖尾特效)
        cur_pos = env._getDroneStateVector(0)[0:3]
        p.addUserDebugLine(
            lineFromXYZ=prev_pos, 
            lineToXYZ=cur_pos, 
            lineColorRGB=[0, 1, 1], # 你的轨迹是青色的 [R=0, G=1, B=1]
            lineWidth=2.5,
            lifeTime=1.5,           # 1.5秒后消散
            physicsClientId=env.CLIENT
        )
        prev_pos = cur_pos

        # 保持仿真帧率 (30Hz)
        time.sleep(1/30)

        # 结算画面
        if terminated or truncated:
            if terminated and reward > 100:
                print("🎯 恭喜！成功抵达目标！")
            else:
                print("💥 挑战失败 (坠毁或超时)！重新开始...")
            
            # 自动重置
            obs, info = env.reset()
            prev_pos = env._getDroneStateVector(0)[0:3]
            
    env.close()



if __name__ == "__main__":
    # manual_control()
    # train()
    test()
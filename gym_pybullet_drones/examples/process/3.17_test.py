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

        # === 新增：动态目标属性 ===
        self.target_anchor = np.zeros(3) # 记录红球出生点 (用于限制活动范围)
        self.target_v = np.zeros(3)      # 红球的三维移动速度
        # === 新增：机动模式与圆周运动参数 ===
        self.target_mode = 0        # 0:X轴, 1:Y轴, 2:Z轴, 3:圆周
        self.target_angle = 0.0     # 圆周运动的当前极角
        self.target_omega = 0.0     # 圆周运动的角速度 (rad/s)
        self.target_radius = 1.5    # 圆周运动的半径

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

        # === 升级：增加 3 维目标相对速度，总计 16 维 ===
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

        # === 核心升级 2：动作空间扩增为 4 维 ===
        #[前后速度(vx), 左右速度(vy), 上下速度(vz), 转向速度(yaw_rate)]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs_raw, info = super().reset(seed=seed, options=options)

        # 随机生成运动的“中心锚点” (防止红球钻地，Z轴基础高度调高点)
        self.target_anchor = np.array([
            np.random.uniform(-5.0, 5.0),
            np.random.uniform(-5.0, 5.0),
            np.random.uniform(0.5, 3.0)
        ])
        
        # 随机抽取本局的运动模式 (0:X直线, 1:Y直线, 2:Z上下, 3:水平圆周)
        self.target_mode = np.random.randint(0, 4)
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
            self.target_omega = np.random.choice([-0.8, 0.8])   # 随机顺时针或逆时针
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
        
        state = self._getDroneStateVector(0)
        self.prev_dist = np.linalg.norm(self.target_pos - state[0:3])

        self.user_input_pos = state[0:3].copy()
        self.current_target_pos = state[0:3].copy()
        self.target_yaw = 0.0 # 每一局重置视角

        if self.GUI:
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

        # ===  新增：获取目标的第一人称相对速度 ===
        local_target_v, _ = p.multiplyTransforms([0,0,0], inv_quat, self.target_v, [0,0,0,1])
        local_target_v = np.array(local_target_v)

        rpy = state[7:10]
        z_height = state[2] 
        
        # 将 local_target_v 也拼接到最后，总计 16 个元素
        return np.concatenate([local_rel_pos, local_vel, rpy,[z_height], local_virt_pos, local_target_v]).astype(np.float32)

    def step(self, action):
        state = self._getDroneStateVector(0)
        dt = 1 / self.CTRL_FREQ

        if self.target_mode in [0, 1, 2]:
            # 直线模式 (X/Y/Z)
            # 更新目标位置: 新位置 = 老位置 + 速度 * 时间
            self.target_pos += self.target_v * dt
            axis = self.target_mode
            
            # 限制移动范围 (偏离锚点 2.0 米则反弹)
            if abs(self.target_pos[axis] - self.target_anchor[axis]) > 2.0:
                self.target_v[axis] *= -1 # 速度反转
                # 强行拉回边界内，防止卡墙穿模
                self.target_pos[axis] = self.target_anchor[axis] + np.sign(self.target_pos[axis] - self.target_anchor[axis]) * 2.0
                
        elif self.target_mode == 3:
            # 圆周模式
            self.target_angle += self.target_omega * dt # 更新极角
            
            # 更新绝对坐标
            self.target_pos[0] = self.target_anchor[0] + self.target_radius * np.cos(self.target_angle)
            self.target_pos[1] = self.target_anchor[1] + self.target_radius * np.sin(self.target_angle)
            
            # 实时更新切线方向的瞬时速度 (这极度重要，无人机会从 local_target_v 里读取这个变化)
            self.target_v[0] = -self.target_radius * self.target_omega * np.sin(self.target_angle)
            self.target_v[1] =  self.target_radius * self.target_omega * np.cos(self.target_angle)
            
        # 防止红球钻入地下 (Z轴托底)
        if self.target_pos[2] < 0.2:
            self.target_pos[2] = 0.2
            if self.target_mode == 2: self.target_v[2] *= -1
    
        # 更新 PyBullet 实体位置
        if self.GUI and self.target_obj_id != -1:
            p.resetBasePositionAndOrientation(self.target_obj_id, self.target_pos,[0, 0, 0, 1])


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
        

        # --- 奖励函数全面进化 (拦截模式) ---

        new_state = self._getDroneStateVector(0)
        new_dist = np.linalg.norm(self.target_pos - new_state[0:3])
        drone_world_vel = new_state[10:13] # 获取无人机当前世界速度
        
        reward = 0 

        # 模块 1：驱动力 (Drive) - 负责让无人机靠近目标
        # 1.1 势能进度奖励 (靠近给正分，远离给负分，绝对主力驱动)
        progress = self.prev_dist - new_dist
        reward_progress = progress * 150.0 

        # 1.2 磁吸诱惑奖励 (距离 < 1.0m 激活，越近分越高)
        reward_proximity = 0.0
        if new_dist < 1.0:
            # 优化：改成线性增加，最大值 0.04。
            # 注意：最大诱惑(0.04)必须小于时间惩罚(0.05)，否则会产生“无限悬停刷分”的 Bug！
            reward_proximity = 0.04 * (1.0 - new_dist) 

        # 模块 2：精准拦截 (Interception) - 负责末端速度匹配
        reward_vel_match = 0.0
        if 0.3 < new_dist < 1.5:
            vel_diff = np.linalg.norm(self.target_v - drone_world_vel)
            # 优化：动态权重。距离 1.5 米时权重为 0，距离 0 米时权重最大(0.1)。
            # 这样避免了跨越 1.0 米边界时奖励突然“断崖式下跌”，动作会更丝滑。
            weight = 0.1 * ((1.5 - new_dist) / 1.5)
            reward_vel_match = -weight * vel_diff

        # 模块 3：第一人称视觉 (Vision Gaze) - 负责机头朝向
        # 在机体坐标系中，正前方是 X 轴 (local_rel[0])
        # 如果 X 轴不是最大正分量，说明无人机没有“看”向红球
        reward_vision = 0.0
        _, inv_quat = p.invertTransform([0,0,0], new_state[3:7])
        local_rel, _ = p.multiplyTransforms([0,0,0], inv_quat, self.target_pos - new_state[0:3],[0,0,0,1])
        
        local_xy_dist = np.linalg.norm(local_rel[0:2])

        if local_xy_dist > 0.2:
            # 计算目标与机头正前方的夹角误差
            cos_yaw_angle = np.clip(local_rel[0] / local_xy_dist, -1.0, 1.0)
            # 距离 1.5 米外，严格要求机头对准(权重1.0)；越靠近，权重越小；最后允许盲抓拦截
            vision_weight = np.clip(new_dist / 1.5, 0.0, 1.0)
            reward_vision = -0.1 * vision_weight * (1.0 - cos_yaw_angle)      # 视野偏离得越多，扣分越多！

        # 模块 4：飞行员素养 (Safety & Smoothness) - 负责安全与姿态
        reward_time = -0.05                                     # 随时间流逝的固定惩罚
        reward_smoothness = -0.02 * np.linalg.norm(action)**2   # 动作抖动惩罚
        
        reward_safety = 0.0
        roll, pitch = new_state[7:9]
        if abs(roll) > 1.0 or abs(pitch) > 1.0:                 # 危险倾角惩罚
            reward_safety = -2.0

        # 模块 5：终止条件 (Terminal) - 负责判定输赢
        reward_terminal = 0.0
        terminated = False

        if new_dist < 0.15:
            reward_terminal = 200.0
            terminated = True
        elif new_state[2] < 0.1 or new_dist > 15.0:
            reward_terminal = -50.0  # 坠毁/飞丢大惩罚
            terminated = True

        # === 汇总所有模块 ===
        reward = (reward_progress + reward_proximity + reward_vel_match + 
                  reward_vision + reward_smoothness + reward_time + 
                  reward_safety + reward_terminal)

        self.prev_dist = new_dist
        
        if (self.step_counter / self.PYB_FREQ) > 20.0:
            truncated = True
            
        return self._computeObs(), reward, terminated, truncated, info

# ---------------------------------------------------------
# 训练与测试设置 
# ---------------------------------------------------------

MODEL_PATH = "drone_vision_v3"
VEC_NORM_PATH = "vec_normalize_vision_v3.pkl"

def train():
    env = DummyVecEnv([lambda: DronePIDEnv(gui=False)])
    
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
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
        model = PPO("MlpPolicy", env, verbose=1, batch_size=256, 
                    learning_rate=3e-4, n_steps=1024, 
                    policy_kwargs=policy_kwargs, tensorboard_log="./ppo_logs/")

    print("开始导航训练 (按下 Ctrl+C 可以安全提前停止并保存)...")
    try:
        model.learn(total_timesteps=200_000, reset_num_timesteps=False) 
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
    print("  按键 [2] : 上帝固定视角 ")
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
    
    obs = env.reset()

    prev_pos = base_env._getDroneStateVector(0)[0:3] # 记录无人机旧位置
    prev_target_pos = base_env.target_pos.copy()     # 记录红球旧位置
    cam_pos = prev_pos.copy()                        # 虚拟云台位置

    camera_mode = 1 # 默认 1: 追尾视角, 2: 上帝视角
    episode_count = 0

    while episode_count < 20: 
        # === 键盘监听：视角切换 ===
        keys = p.getKeyboardEvents(physicsClientId=base_env.CLIENT)
        if ord('1') in keys and keys[ord('1')] & p.KEY_WAS_TRIGGERED:
            camera_mode = 1
            print("切换至：智能追尾视角")
        if ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED:
            camera_mode = 2
            print("切换至：上帝固定全景视角")

        action, _ = model.predict(obs, deterministic=True) # AI 思考并输出动作
        obs, reward, done, info = env.step(action)         #环境执行动作

        cur_pos = base_env._getDroneStateVector(0)[0:3]
        cur_target_pos = base_env.target_pos.copy()
        
        if not done[0]:
            # === 画线 1：无人机飞行轨迹 (青色) ===
            p.addUserDebugLine(
                lineFromXYZ=prev_pos, lineToXYZ=cur_pos, 
                lineColorRGB=[0, 1, 1], lineWidth=2.5, lifeTime=1.5, 
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
            

            # === 运镜控制 ===
            if camera_mode == 1:
                # 模式 1：丝滑追尾视角
                cam_pos = cam_pos * 0.95 + cur_pos * 0.05 # 吸收高频震动 (0.95保留历史，0.05追踪当前)
                smooth_yaw = np.degrees(base_env.target_yaw)
                p.resetDebugVisualizerCamera(
                    cameraDistance=2.0,
                    cameraYaw=smooth_yaw - 90, 
                    cameraPitch=-20,
                    cameraTargetPosition=cam_pos,
                    physicsClientId=base_env.CLIENT
                )
            elif camera_mode == 2:
                # 模式 2：上帝俯视视角 (固定在 8 米高空，俯视整个 10x10 米的场地)
                p.resetDebugVisualizerCamera(
                    cameraDistance=8.0,
                    cameraYaw=0, 
                    cameraPitch=-89.9, # 不能设置绝对 -90 度，PyBullet会有万向节死锁问题
                    cameraTargetPosition=[0, 0, 1.0],
                    physicsClientId=base_env.CLIENT
                )
           
        time.sleep(1/30) 
        
        if done[0]:
            episode_count += 1
            print(f"第 {episode_count} 轮测试结束！")

            prev_pos = base_env._getDroneStateVector(0)[0:3]
            prev_target_pos = base_env.target_pos.copy()
            cam_pos = prev_pos.copy() # 重置云台位置
            
    env.close()

def manual_control():
    print("==================================")
    print("开启第一人称 (FPV) 手动驾驶模式！")
    print("操作说明 (请确保鼠标点击激活了 PyBullet 仿真窗口)：")
    print("  [U] / [J] : 前进 / 后退 (本地 X 轴)")
    print("  [H] / [K] : 转向左 / 转向右 (Yaw 偏航控制!)")
    print("  [↑] / [↓] : 升高 / 降低 (Z轴)")
    print("  [←] / [→] : 侧飞左 / 侧飞右 (本地 Y 轴)")
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
        if ord('u') in keys and keys[ord('u')] & p.KEY_IS_DOWN: action[0] = 1.0
        if ord('j') in keys and keys[ord('j')] & p.KEY_IS_DOWN: action[0] = -1.0
        
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN: action[1] = 1.0
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN: action[1] = -1.0
            
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN: action[2] = 1.0
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN: action[2] = -1.0
            
        if ord('h') in keys and keys[ord('h')] & p.KEY_IS_DOWN: action[3] = 1.0 # 视角左转
        if ord('k') in keys and keys[ord('k')] & p.KEY_IS_DOWN: action[3] = -1.0 # 视角右转

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
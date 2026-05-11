import os
import time
import numpy as np
import pybullet as p
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

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

        # 动作重复次数（k次）
        self.frame_skip = 10
        
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=np.array([[0, 0, 1.0]]),
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=60,
            gui=gui,
        )
        self.EPISODE_LEN_SEC = 10 
        self.pid = DSLPIDControl(drone_model=DroneModel.CF2X)

        self.SMOOTH_FACTOR = 0.1
        self.user_input_pos = np.zeros(3)      
        self.current_target_pos = np.zeros(3)  
        
        # 加入 Yaw 控制变量 
        self.target_yaw = 0.0 

        # === 升级：增加 3 维目标相对速度，总计 16 维 ===
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

        self.action_space = gym.spaces.Discrete(11)

        # 建立 NASA BFM 到物理控制量的映射: {动作编号 : (切向过载 n_x, 法向过载 n_n, 滚转角 mu)}
        self.bfm_action_mapping = {
            0:  ( 0,  1,  0.0),            # a1: 匀速直飞
            1:  ( 2,  1,  0.0),            # a2: 加速直飞
            2:  (-2,  1,  0.0),            # a3: 减速直飞
            3:  ( 0,  8,  0.0),            # a4: 跃升
            4:  ( 0, -8,  0.0),            # a5: 俯冲
            5:  ( 0,  8,  np.pi / 3.0),    # a6: 左转跃升
            6:  ( 0, -8, -np.pi / 3.0),    # a7: 右转俯冲
            7:  ( 0,  8, -np.pi / 3.0),    # a8: 右转跃升
            8:  ( 0, -8,  np.pi / 3.0),    # a9: 左转俯冲
            9:  ( 0,  2, -np.pi / 3.0),    # a10: 右转
            10: ( 0,  2,  np.pi / 3.0)     # a11: 左转
        }
        '''
        # 建立 11 种离散动作到连续控制指令 [前后(x), 左右(y), 上下(z), 转向(yaw)] 的映射字典
        # 数值域为 [-1, 1]，即原有连续空间的控制比例
        self.bfm_action_mapping = {
            0:  np.array([ 0.4, 0.0,  0.0,  0.0]), # a1: 匀速直飞 (保持中等速度巡航)
            1:  np.array([ 1.0, 0.0,  0.0,  0.0]), # a2: 加速直飞 (全速追击)
            2:  np.array([-0.5, 0.0,  0.0,  0.0]), # a3: 减速直飞 (向后刹车)
            3:  np.array([ 0.7, 0.0,  0.8,  0.0]), # a4: 跃升 (边前进边拉高)
            4:  np.array([ 0.7, 0.0, -0.8,  0.0]), # a5: 俯冲 (边前进边下降)
            5:  np.array([ 0.7, 0.0,  0.8,  1.0]), # a6: 左转跃升 (yaw>0 为左转)
            6:  np.array([ 0.7, 0.0, -0.8, -1.0]), # a7: 右转俯冲
            7:  np.array([ 0.7, 0.0,  0.8, -1.0]), # a8: 右转跃升
            8:  np.array([ 0.7, 0.0, -0.8,  1.0]), # a9: 左转俯冲
            9:  np.array([ 0.7, 0.0,  0.0, -1.0]), # a10: 右转
            10: np.array([ 0.7, 0.0,  0.0,  1.0])  # a11: 左转
        }

        '''
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

        # 传感器视觉投影 (World Frame -> Body Frame) 
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

        rpy = state[7:10]   # 自身姿态
        z_height = state[2] # 提取绝对高度
        
        # 将 local_target_v 也拼接到最后，总计 16 个元素
        return np.concatenate([local_rel_pos, local_vel, rpy,[z_height], local_virt_pos, local_target_v]).astype(np.float32)

    def step(self, action):
        state = self._getDroneStateVector(0)
        dt = 1 / self.CTRL_FREQ

        # 解码离散动作
        action_int = int(action)
        n_x, n_n, mu = self.bfm_action_mapping[action_int]

        # 初始化累加奖励和终止状态
        total_reward = 0.0
        terminated = False
        truncated = False
        info_pyb = {} # 用于接收底层物理引擎的 info

        # 动作重复循环 (Frame Skip)
        for _ in range(self.frame_skip):

            # 将目标的运动逻辑移入内部循环，确保时间流逝与无人机完全同步
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


            # 将战斗机过载指令转换为四旋翼局部速度 (物理等效映射)
            # quat = self._getDroneStateVector(0)[3:7]

            #   - 切向过载 nx 转为前进速度 (默认基础前飞速度1.0m/s，最大2.0m/s，最小0m/s)
            vx = 1.0 + n_x * 0.5 
            #   - BFM库没有平移侧飞概念，严格遵循空战气动，强制为0
            vy = 0.0 
            #   - 法向过载 nn 与滚转角 mu 的垂直分量转化为爬升/俯冲速度 (平飞时nn*cos(0)=1，抵消重力，vz=0)
            vz = (n_n * np.cos(mu) - 1.0) * 0.3
            
            # 法向过载 nn 与滚转角 mu 的水平分量转化为偏航转向角速度
            yaw_rate = n_n * np.sin(mu) * 0.25
            
            target_vel_local = np.array([vx, vy, vz])
            self.target_yaw += yaw_rate * dt
            # 限制 target_yaw 在 [-pi, pi] 之间，防止数值爆炸
            self.target_yaw = (self.target_yaw + np.pi) % (2 * np.pi) - np.pi
            
            # 生成一个没有俯仰(Pitch)和滚转(Roll)的纯净偏航姿态
            pilot_quat = p.getQuaternionFromEuler([0, 0, self.target_yaw])

            # 将第一人称飞行指令转回世界坐标，用于移动我们的“虚拟目标点”
            vel_world, _ = p.multiplyTransforms([0,0,0], pilot_quat, target_vel_local, [0,0,0,1])
            vel_world = np.array(vel_world)

            self.user_input_pos += vel_world * dt

            if self.user_input_pos[2] < 0.5:
                self.user_input_pos[2] = 0.5
            
            self.current_target_pos = self.current_target_pos * (1 - self.SMOOTH_FACTOR) + self.user_input_pos * self.SMOOTH_FACTOR
            
            # 底层 PID 控制
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
            
            obs_raw, _, _, truncated, info_pyb = super().step(rpm.reshape(1, 4))

            if self.GUI:
                # 每步控制频率是 30Hz，强制程序休眠 1/30 秒，实现 1:1 真实物理平滑渲染
                time.sleep(1 / self.CTRL_FREQ) 
            
            # 优化版 BFM 空战截击奖励函数
            # ==========================================
            
            # 1. 计算相对位置与距离
            cur_pos = self._getDroneStateVector(0)[0:3]  
            rel_pos = self.target_pos - cur_pos
            dist = np.linalg.norm(rel_pos)
            
            # 2. 计算机头指向与目标的夹角 (即论文中的 ATA - 天线前置角)
            # 获取无人机当前的姿态四元数转旋转矩阵
            _, orientation = self._getDroneStateVector(0)[0:3], self._getDroneStateVector(0)[3:7]
            rot_mat = p.getMatrixFromQuaternion(orientation)
            # 假设机头沿着局部 X 轴正方向 (根据您的具体模型可能需要改成Y轴 rot_mat[1], [4], [7])
            forward_vector = np.array([rot_mat[0], rot_mat[3], rot_mat[6]]) 
            
            # 归一化目标方向向量
            target_dir = rel_pos / (dist + 1e-6)
            
            # 计算机头与目标的余弦相似度 (-1 到 1，1表示正对目标)
            cos_theta = np.clip(np.dot(forward_vector, target_dir), -1.0, 1.0)
            # 转换为角度差 (弧度, 0 表示正对，pi 表示背对)
            ata_angle = np.arccos(cos_theta) 

            # ------------------------------------------
            # 奖励项 A：姿态对准奖励 (Tracking Reward) [极其关键]
            # 鼓励无人机始终把机头对准目标，像战斗机雷达锁定一样
            # 如果机头偏差小于 30度(约0.5弧度)，给予正奖励；背对则惩罚
            reward_tracking = 1.0 - (ata_angle / (np.pi / 2)) 
            reward_tracking = np.clip(reward_tracking, -1.0, 1.0) * 2.0

            # ------------------------------------------
            # 奖励项 B：距离逼近奖励 (Approach Reward)
            # 只有在机头大致对准目标时，靠近才有意义；如果是背对目标越飞越远，惩罚
            if ata_angle < np.pi / 2:
                # 正在朝目标飞，距离越近奖励越大
                reward_distance = 2.0 / (dist + 1.0) 
            else:
                # 背对目标，且距离变远，给予惩罚逼迫它转弯
                reward_distance = -0.05 * dist 

            # ------------------------------------------
            # 奖励项 C：过载与抖动惩罚 (Smoothness Penalty)
            # 离散动作极易产生 10(左转) 和 9(右转) 交替输出的“帕金森”飞行
            # 对非 0/1/2 (直飞) 的大过载机动给予微小惩罚，鼓励在能直飞时就直飞
            reward_action = -0.01 * (abs(n_x) + abs(n_n - 1.0) + abs(mu))

            # ------------------------------------------
            # 奖励项 D：稀疏任务奖励 (Terminal Reward)
            reward_terminal = 0.0
            terminated = False
            
            # 成功截击：距离极近且机头对准 (满足导弹发射条件)
            if dist < 0.5 and ata_angle < 0.3:
                reward_terminal = 100.0  # 给予巨大奖励
                terminated = True
                print("🚀 成功锁定并截击目标！")
                
            # 失败惩罚：飞出作战空域 (比如离中心太远)
            elif dist > 15.0 or cur_pos[2] < 0.2: # 飞太远或坠毁
                reward_terminal = -800.0
                terminated = True

            if (self.step_counter / self.PYB_FREQ) > self.EPISODE_LEN_SEC:
                truncated = True

            # === 总奖励 ===
            reward = reward_tracking + reward_distance + reward_action + reward_terminal
            total_reward += reward 

            if terminated or truncated:
                break

        # 重新计算循环跳跃结束后的最终环境观测快照
        final_obs = self._computeObs()
        
        # 组装 info 字典（保留底层环境自带的info，同时塞入我们的专家机动动作数据）
        info = info_pyb
        info["BFM_action"] = {
            "action_id": action_int,
            "n_x (g)": n_x, 
            "n_n (g)": n_n, 
            "roll (rad)": round(mu, 2)
        }    
        
        return final_obs, total_reward, terminated, truncated, info

# ---------------------------------------------------------
# 训练与测试设置 
# ---------------------------------------------------------

MODEL_PATH = "drone_vision_v6"
VEC_NORM_PATH = "vec_normalize_vision_v6.pkl"

def train():
    env = DummyVecEnv([lambda: Monitor(DronePIDEnv(gui=False))])
    
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
        model.learn(total_timesteps=500_000, reset_num_timesteps=False,tb_log_name="run_3.31_BFM") 
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
    print("  按键 [2] : 上帝固定视角 （俯视全景）")
    print("  按键 [3] : 目标迎击视角 ")
    print("  按键 [4] : 目标固定广角 (以红球为中心，拉远固定视角)")
    print("  按键 [5] : 上帝侧面平视视角 (固定在场地一侧，看高度拉扯)")
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

    camera_mode = 4 
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
        if ord('3') in keys and keys[ord('3')] & p.KEY_WAS_TRIGGERED:
            camera_mode = 3
            print("切换至：目标迎击视角")
        if ord('4') in keys and keys[ord('4')] & p.KEY_WAS_TRIGGERED:
            camera_mode = 4
            print("切换至：目标固定广角视角")
        if ord('5') in keys and keys[ord('5')] & p.KEY_WAS_TRIGGERED:
            camera_mode = 5
            print("切换至：上帝侧面平视视角")
        action, _ = model.predict(obs, deterministic=True) # AI 思考并输出动作

        real_cur_pos = base_env._getDroneStateVector(0)[0:3]
        print(f"AI 打算执行动作: [{action[0] if isinstance(action, np.ndarray) else action}], 执行前绝对高度: {real_cur_pos[2]:.2f}m")
    
        obs, reward, done, info = env.step(action)         #环境执行动作

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
            elif camera_mode == 3:
                # 模式 3： 目标迎击视角
                # 1. 计算无人机和红球的距离
                dist = np.linalg.norm(cur_pos - cur_target_pos)
                
                # 2. 计算无人机相对于红球的方位角
                dx = cur_pos[0] - cur_target_pos[0]
                dy = cur_pos[1] - cur_target_pos[1]
                drone_angle = np.degrees(np.arctan2(dy, dx))

                # 3. 动态运镜魔法
                p.resetDebugVisualizerCamera(
                    # 距离动态缩放：距离远时镜头拉宽，距离近时镜头逼近 (最小保持 1.5 米距离)
                    cameraDistance=max(1.5, dist * 0.8), 
                    
                    # 偏航角：站在红球偏左 45 度的位置看向无人机，形成极具电影感的 3/4 侧方位迎面压迫感
                    cameraYaw=drone_angle - 45,          
                    
                    # 稍微俯视，确保即使无人机飞得很高也能留在画面内
                    cameraPitch=-20,                     
                    
                    # 焦点死死锁住红球！
                    cameraTargetPosition=cur_target_pos, 
                    physicsClientId=base_env.CLIENT
                )
            elif camera_mode == 4:
                # 模式 4：目标固定广角视角 (红球中心，机位锁定在固定角度)
                # 无论红球怎么跑，镜头始终在它右上方的固定 4 米处俯视它，视野极其开阔！
                p.resetDebugVisualizerCamera(
                    cameraDistance=4.0,          # 距离拉远，涵盖整个拦截空域
                    cameraYaw=45,                # 固定偏航角
                    cameraPitch=-30,             # 固定俯仰角
                    cameraTargetPosition=cur_target_pos, # 焦点锁死红球
                    physicsClientId=base_env.CLIENT
                )
            elif camera_mode == 5:
                # 模式 5：上帝侧面平视视角
                # 获取红球这局游戏的“初始锚点高度”，作为镜头的水平中心
                anchor_z = base_env.target_anchor[2] 
                
                p.resetDebugVisualizerCamera(
                    # 距离拉到 10 米外，足以把 10x10 米的场地横截面全部收进屏幕
                    cameraDistance=10.0, 
                    
                    # 90度代表站在 Y 轴的极远端，沿着 X 轴平视看过去
                    cameraYaw=90,        
                    
                    # -5度的微小俯角。绝对平视(0度)容易看不清地面的纵深，微微向下 5 度立体感最好
                    cameraPitch=-5,      
                    
                    # 镜头中心死死锁在场地正中心[0, 0] 的“目标初始高度”上
                    cameraTargetPosition=[0, 0, anchor_z], 
                    
                    physicsClientId=base_env.CLIENT
                )
        
        if done[0]:
            episode_count += 1
            print(f"第 {episode_count} 轮测试结束！")

            prev_pos = base_env._getDroneStateVector(0)[0:3]
            prev_target_pos = base_env.target_pos.copy()
            cam_pos = prev_pos.copy() # 重置云台位置
            
    env.close()

def manual_control():
    print("==================================")
    print("开启第一人称离散动作 (BFM) 手动驾驶模式！")
    print("操作说明 (经典飞行器键位，点按生效)：")
    print("  [1] : 匀速直飞 (松开按键默认恢复此状态)")
    print("  [2] : 加速直飞    [3] : 减速直飞/悬停")
    print("  [W] : 跃升        [S] : 俯冲")
    print("  [A] : 左平转      [D] : 右平转")
    print("[Q] : 左转跃升    [E] : 右转跃升")
    print("[Z] : 左转俯冲    [C] : 右转俯冲")
    print("  [R] : 重置环境    [ESC]: 退出")
    print("==================================")

    # ====================== 新增：动作优先级与可调持续时间 ======================
    # 优先级（数字越大优先级越高）：组合机动 > 姿态机动 > 速度机动
    PRIORITY = {
        0:  0,   # 默认直飞
        1:  1,   # 加速
        2:  1,   # 减速
        3:  2,   # 跃升
        4:  2,   # 俯冲
        9:  2,   # 右平转
        10: 2,   # 左平转
        5:  3,   # 左转跃升
        7:  3,   # 右转跃升
        8:  3,   # 左转俯冲
        6:  3,   # 右转俯冲
    }

    # 默认最小持续时间（单位：秒）——基于真实空战 BFM 经验值
    # 你可以在这里自由修改每个动作的具体时长
    MIN_DURATIONS = {
        1:  0.4,   # 加速直飞（快速油门响应）
        2:  0.5,   # 减速/悬停
        3:  0.6,   # 跃升
        4:  0.6,   # 俯冲
        9:  0.5,   # 右平转
        10: 0.5,   # 左平转
        5:  0.8,   # 左转跃升（组合机动，需要更长时间协调）
        7:  0.8,   # 右转跃升
        8:  0.8,   # 左转俯冲
        6:  0.8,   # 右转俯冲
    }

    # 动作 → 对应键盘码（用于“按住延长”功能）
    ACTION_TO_KEY = {
        1: ord('2'), 2: ord('3'),
        3: ord('w'), 4: ord('s'),
        5: ord('q'), 6: ord('c'),
        7: ord('e'), 8: ord('z'),
        9: ord('d'), 10: ord('a'),
    }
    # =========================================================================

    env = DronePIDEnv(gui=True)
    env.frame_skip = 1
    env.EPISODE_LEN_SEC = 60

    obs, info = env.reset()
    prev_pos = env._getDroneStateVector(0)[0:3]
    cam_pos = prev_pos.copy()

    # 新增持久变量
    current_action = 0
    action_remaining_time = 0.0
    last_time = time.time()

    while True:
        keys = p.getKeyboardEvents()
        dt = time.time() - last_time
        last_time = time.time()

        # 更新剩余强制执行时间
        if action_remaining_time > 0:
            action_remaining_time -= dt
            if action_remaining_time < 0:
                action_remaining_time = 0

        # 退出与重置
        if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
            break

        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            obs, info = env.reset()
            prev_pos = env._getDroneStateVector(0)[0:3]
            cam_pos = prev_pos.copy()
            current_action = 0
            action_remaining_time = 0.0
            continue

        # === 收集本次按下的新动作（使用 WAS_TRIGGERED，支持同时按多个键）===
        triggered_actions = []
        if ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED:
            triggered_actions.append(1)
        if ord('3') in keys and keys[ord('3')] & p.KEY_WAS_TRIGGERED:
            triggered_actions.append(2)
        if ord('w') in keys and keys[ord('w')] & p.KEY_WAS_TRIGGERED:
            triggered_actions.append(3)
        if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
            triggered_actions.append(4)
        if ord('a') in keys and keys[ord('a')] & p.KEY_WAS_TRIGGERED:
            triggered_actions.append(10)
        if ord('d') in keys and keys[ord('d')] & p.KEY_WAS_TRIGGERED:
            triggered_actions.append(9)
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            triggered_actions.append(5)
        if ord('e') in keys and keys[ord('e')] & p.KEY_WAS_TRIGGERED:
            triggered_actions.append(7)
        if ord('z') in keys and keys[ord('z')] & p.KEY_WAS_TRIGGERED:
            triggered_actions.append(8)
        if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
            triggered_actions.append(6)

        # 如果有新按键触发，按优先级选择最高的一个
        if triggered_actions:
            new_action = max(triggered_actions, key=lambda a: PRIORITY.get(a, 0))
            current_action = new_action
            action_remaining_time = MIN_DURATIONS.get(new_action, 0.5)

        # 按住延长机制：如果当前动作对应的键仍然按住，就把剩余时间刷新到至少默认时长
        if current_action != 0:
            key_code = ACTION_TO_KEY.get(current_action)
            if key_code and key_code in keys and keys[key_code] & p.KEY_IS_DOWN:
                action_remaining_time = max(action_remaining_time,
                                            MIN_DURATIONS.get(current_action, 0.5))

        # 决定本次真正要执行的动作
        if action_remaining_time > 0:
            action = current_action
        else:
            action = 0   # 默认匀速直飞

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(np.array([action]))

        # 绘制轨迹、目标点、相机（保持原逻辑）
        cur_pos = env._getDroneStateVector(0)[0:3]
        p.addUserDebugLine(prev_pos, cur_pos, [0, 1, 1], 2.5, 1.5, physicsClientId=env.CLIENT)
        prev_pos = cur_pos

        carrot = env.current_target_pos
        p.addUserDebugLine(carrot-[0.1,0,0], carrot+[0.1,0,0], [1,0,1], 2, 0.05, physicsClientId=env.CLIENT)
        p.addUserDebugLine(carrot-[0,0.1,0], carrot+[0,0.1,0], [1,0,1], 2, 0.05, physicsClientId=env.CLIENT)

        cam_pos = cam_pos * 0.8 + cur_pos * 0.2
        smooth_yaw = np.degrees(env.target_yaw)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5, cameraYaw=smooth_yaw - 90,
            cameraPitch=-15, cameraTargetPosition=cam_pos
        )

        if terminated or truncated:
            obs, info = env.reset()
            prev_pos = env._getDroneStateVector(0)[0:3]
            cam_pos = prev_pos.copy()
            current_action = 0
            action_remaining_time = 0.0

    env.close()

if __name__ == "__main__":
    manual_control()
    # train()
    # test() 
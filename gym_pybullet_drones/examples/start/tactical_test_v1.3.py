import time
import numpy as np
import pybullet as p
import matplotlib.colors as mcolors
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

def run_swarm_test():
    print("[INFO] 初始化 多无人机蜂群环境...")

    # --- 1. 配置参数 ---
    NUM_DRONES = 12
    RADIUS = 1.0  # 编队半径
    
    # 计算初始位置：在地面围成一个圆
    # 这样避免起飞时挤在一起碰撞
    INIT_XYZS = []
    for i in range(NUM_DRONES):
        angle = i * (2 * np.pi / NUM_DRONES)
        x = RADIUS * np.cos(angle)
        y = RADIUS * np.sin(angle)
        INIT_XYZS.append([x, y, 0.1]) # z=0.1 从地面起飞
    
    INIT_XYZS = np.array(INIT_XYZS)
    INIT_RPYS = np.zeros((NUM_DRONES, 3)) # 初始姿态全为0

    # --- 2. 环境配置 ---
    env = CtrlAviary(drone_model=DroneModel.CF2X,
                     num_drones=NUM_DRONES,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=Physics.PYB,
                     pyb_freq=240,
                     ctrl_freq=240,
                     gui=True,
                     record=False)

    # 生成彩虹色轨迹颜色 (RGB)
    # 使用 matplotlib 均匀生成 NUM_DRONES 种颜色
    hsv_colors = [((i / NUM_DRONES), 1.0, 1.0) for i in range(NUM_DRONES)]
    rgb_colors = [mcolors.hsv_to_rgb(c) for c in hsv_colors]

    # 初始化 NUM_DRONES 个 PID 控制器
    controllers = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(NUM_DRONES)]

    obs, info = env.reset(seed=42)
    
    # 初始摄像头位置
    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

    print("------------------------------------------------")
    print(f"[READY] {NUM_DRONES}架无人机已就位。")
    print("[ACTION] 它们将形成阵列并螺旋上升。")
    print("[VIEW] 摄像头将自动锁定蜂群中心。")
    print("------------------------------------------------")
    input("按回车键开始表演 >> ")

    CTRL_FREQ = 240
    TOTAL_STEPS = 10000 
    START_TIME = time.time()
    action = np.zeros((NUM_DRONES, 4))
    
    # 记录上一帧位置用于画线
    last_pos = obs[:, 0:3] # Shape: (5, 3)

    for i in range(TOTAL_STEPS):
        current_time = i / CTRL_FREQ
        
        # --- 计算蜂群中心 (用于摄像头追踪) ---
        # axis=0 表示对所有飞机的坐标取平均值
        center_pos = np.mean(obs[:, 0:3], axis=0) 
        
        # 实时更新摄像头看着中心点，并在 z 轴稍微抬高一点视角
        '''
        p.resetDebugVisualizerCamera(
            cameraDistance=2.5,
            cameraYaw=(current_time * 10) - 45, # 让摄像头也慢慢旋转，制造电影感！
            cameraPitch=-30,
            cameraTargetPosition=center_pos
        )
'''

        # --- 控制循环 ---
        for j in range(NUM_DRONES):
            # 战术动作：螺旋上升
            # 基础角度 + 时间增量 = 旋转
            base_angle = j * (2 * np.pi / NUM_DRONES)
            rot_speed = 0.5
            current_angle = base_angle + current_time * rot_speed
            
            target_x = RADIUS * np.cos(current_angle)
            target_y = RADIUS * np.sin(current_angle)
            
            # 高度随时间上升，起飞后保持在 1.0 + 浮动
            target_z = 1.0 + (current_time * 0.1) 
            
            # 计算 PID
            action[j], _, _ = controllers[j].computeControlFromState(
                control_timestep=1/CTRL_FREQ,
                state=obs[j],
                target_pos=np.array([target_x, target_y, target_z]),
                target_vel=np.zeros(3)
            )

        # --- 执行 ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- 画轨迹 ---
        if i % 10 == 0:
            for j in range(NUM_DRONES):
                p.addUserDebugLine(last_pos[j], obs[j][0:3], lineColorRGB=rgb_colors[j], lifeTime=5, lineWidth=2)
            last_pos = obs[:, 0:3]

        # --- 简洁输出 (只打印中心高度) ---
        if i % 240 == 0:
            print(f"时间: {current_time:.1f}s | 蜂群中心高度: {center_pos[2]:.2f}m")

        # --- 同步 ---
        elapsed = time.time() - START_TIME
        if elapsed < (i + 1) / CTRL_FREQ:
            time.sleep((i + 1) / CTRL_FREQ - elapsed)

    env.close()

if __name__ == "__main__":
    run_swarm_test()
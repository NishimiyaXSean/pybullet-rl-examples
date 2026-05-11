import time
import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

def run_leader_follower_test():
    print("[INFO] 初始化【领机-僚机】编队任务...")

    # --- 1. 初始位置 ---
    # 领机在原点上方，僚机在领机左后方，避免起飞碰撞
    INIT_XYZS = np.array([
        [0.0, 0.0, 0.0],    # 领机 (Drone 0)
        [-1.0, -1.0, 0]   # 僚机 (Drone 1)
    ])
    INIT_RPYS = np.array([[0,0,0], [0,0,0]])

    # --- 2. 环境配置 (保持 240Hz 稳定配置) ---
    env = CtrlAviary(drone_model=DroneModel.CF2X,
                     num_drones=2,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=Physics.PYB,
                     pyb_freq=240,
                     ctrl_freq=240,
                     gui=True,
                     record=False)

    # 调整视角，俯瞰战场
    p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])

    # 初始化控制器
    controllers = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]

    obs, info = env.reset(seed=42)
    
    print("------------------------------------------------")
    print("[READY] 战术场景：8字盘旋 + 编队跟随")
    print("[VISUAL] 红色线 = 领机轨迹 | 绿色线 = 僚机轨迹")
    print("[ACTION] 按回车键开始任务...")
    print("------------------------------------------------")
    input()

    CTRL_FREQ = 240
    TOTAL_STEPS = 5000  
    START_TIME = time.time()
    action = np.zeros((2, 4))
    
    # 用于画轨迹的变量
    last_pos_0 = obs[0][0:3]
    last_pos_1 = obs[1][0:3]

    for i in range(TOTAL_STEPS):
        current_time = i / CTRL_FREQ
        
        # ================== 战术指令生成 ==================
        
        # --- 角色1：领机 (Drone 0) 飞 "8" 字 ---
        # 这里的数学公式是 Lemniscate of Bernoulli (伯努利双纽线) 的参数方程变种
        scale = 1.5  # 8字的大小
        speed = 0.5  # 飞行速度
        t = current_time * speed
        
        target_x_0 = scale * np.sin(t)
        target_y_0 = scale * np.sin(t) * np.cos(t)
        target_z_0 = 1.0  # 保持高度
        
        target_pos_0 = np.array([target_x_0, target_y_0, target_z_0])
        
        # --- 角色2：僚机 (Drone 1) 动态跟随 ---
        # 策略：获取领机当前位置，设定目标为领机位置的 "后方" 和 "上方"
        # 简单的跟随偏移量
        follow_offset = np.array([-0.3, -0.3, 0.2]) 
        
        # 僚机的目标 = 领机当前真实位置 (obs[0]) + 偏移量
        # 注意：这里是 obs[0] (领机实际位置)，而不是 target_pos_0 (领机目标位置)，更符合真实的视觉跟随逻辑
        leader_real_pos = obs[0][0:3]
        target_pos_1 = leader_real_pos + follow_offset

        # ================== PID 计算 ==================
        
        # 领机控制
        action[0], _, _ = controllers[0].computeControlFromState(
            control_timestep=1/CTRL_FREQ,
            state=obs[0],
            target_pos=target_pos_0,
            target_vel=np.zeros(3)
        )
        
        # 僚机控制
        action[1], _, _ = controllers[1].computeControlFromState(
            control_timestep=1/CTRL_FREQ,
            state=obs[1],
            target_pos=target_pos_1,
            target_vel=np.zeros(3)
        )

        # ================== 执行与渲染 ==================
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- 视觉特效：画轨迹线 ---
        # 每 10 步画一段线，避免太密集吃内存
        if i % 10 == 0:
            curr_pos_0 = obs[0][0:3]
            curr_pos_1 = obs[1][0:3]
            
            # 领机轨迹 (红色)
            p.addUserDebugLine(last_pos_0, curr_pos_0, lineColorRGB=[1, 0, 0], lifeTime=10, lineWidth=2)
            # 僚机轨迹 (绿色)
            p.addUserDebugLine(last_pos_1, curr_pos_1, lineColorRGB=[0, 1, 0], lifeTime=10, lineWidth=2)
            
            last_pos_0 = curr_pos_0
            last_pos_1 = curr_pos_1

        # --- 朴实无华的控制台输出 (每秒一次) ---
        if i % 240 == 0:
            dist = np.linalg.norm(obs[0][0:3] - obs[1][0:3])
            print(f"时间: {current_time:.1f}s | 领机位置: {obs[0][0:3].round(2)} | 僚机距离领机: {dist:.2f}米")

        # --- 同步时间 ---
        elapsed = time.time() - START_TIME
        if elapsed < (i + 1) / CTRL_FREQ:
            time.sleep((i + 1) / CTRL_FREQ - elapsed)

    env.close()

if __name__ == "__main__":
    run_leader_follower_test()
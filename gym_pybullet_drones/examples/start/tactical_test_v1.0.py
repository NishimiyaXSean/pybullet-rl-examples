import time
import math
import sys
import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
import pybullet as p 

def run_tactical_test():
    print("[INFO] 初始化双机战术环境...")

    # 1.定义初始位置
    # 飞机0: 对应圆周运动的起点 (x=1, y=0, z=1)
    # 飞机1: 对应悬停点的水平位置 (x=0, y=0.0, z=1)
    # 这样飞机一起飞就在目标附近，不需要剧烈机动，避免触地侧翻
    INIT_XYZS = np.array([
        [1.0, 0.0, 1.0], 
        [0.0, 0.0, 1.0]
    ])
    
    # 定义初始姿态 (全部水平)
    INIT_RPYS = np.array([
        [0, 0, 0],
        [0, 0, 0]
    ])
    
    # 2.配置环境
    env = CtrlAviary(drone_model=DroneModel.CF2X,
                     num_drones=2,
                     neighbourhood_radius=100,
                     initial_xyzs=INIT_XYZS, 
                     initial_rpys=INIT_RPYS,
                     physics=Physics.PYB,
                     pyb_freq=240,
                     ctrl_freq=240,
                     gui=True,
                     record=False)

    # 调整摄像头角度，看清楚两架飞机
    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])
    
    # 3.初始化 PID 控制器
    controllers = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]

    # 4.环境重置
    obs, info = env.reset(seed=42)
    env.render()  # 先渲染出一帧画面
    
    print("------------------------------------------------")
    print("[READY] GUI窗口应该已经弹出。")
    print("[ACTION] 请调整窗口位置，然后 按回车键 开始飞行...")
    print("------------------------------------------------")
    input() 

    # 5. 模拟循环参数
    CTRL_FREQ = 240
    TOTAL_STEPS = 10000
    START_TIME = time.time() # 记录开始时间
    action = np.zeros((2, 4)) # 初始化动作数组

    for i in range(TOTAL_STEPS):
        
        #当前模拟时间
        current_time = i / CTRL_FREQ
        
        # --- 飞机0：画圆 (半径1m) ---
        # 目标位置
        target_pos_0 = np.array([
            1.0 * np.cos(current_time), 
            1.0 * np.sin(current_time), 
            1.0
        ])
        # 计算 PID
        action[0], _, _ = controllers[0].computeControlFromState(
            control_timestep=1/CTRL_FREQ,
            state=obs[0],
            target_pos=target_pos_0,
            target_vel=np.zeros(3)
        )

        # --- 飞机1：原地上下浮动 ---
        # 目标位置 (X,Y 固定，Z 变化)
        target_pos_1 = np.array([
            0.0, 
            0.0, 
            1.0 + 0.3 * np.sin(current_time * 2) # 减小一点幅度更稳
        ])
        action[1], _, _ = controllers[1].computeControlFromState(
            control_timestep=1/CTRL_FREQ,
            state=obs[1],
            target_pos=target_pos_1,
            target_vel=np.zeros(3)
        )

        # 6. 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 7. 渲染
        env.render()
        
        # --- 战术数据分析 ---
        if i % 120 == 0: # 每120步打印一次，避免刷屏
            # obs[0][0:3] 是飞机0的 x,y,z
            # obs[1][0:3] 是飞机1的 x,y,z
            pos0 = obs[0][0:3]
            pos1 = obs[1][0:3]
            # 计算欧几里得距离
            distance = np.linalg.norm(pos0 - pos1)
            # 使用 sys.stdout 强制单行刷新
            # sys.stdout.write(f"\rStep:{i:04d} | 机0坐标:({pos0[0]:.1f},{pos0[1]:.1f}) | 机1高度:{pos1[2]:.2f}m | 距离:{distance:.2f}m   ")
            # sys.stdout.flush()
            print(f"\rStep:{i:04d} | 机0坐标:({pos0[0]:.1f},{pos0[1]:.1f}) | 机1高度:{pos1[2]:.2f}m | 距离:{distance:.2f}m   ")


        # 手动同步时间,为了让画面看起来是正常速度，而不是快进
        elapsed = time.time() - START_TIME
        if elapsed < (i + 1) / CTRL_FREQ:
            time.sleep((i + 1) / CTRL_FREQ - elapsed)


    print("\n[INFO] 飞行结束，3秒后关闭窗口...")
    time.sleep(3)
    env.close()

if __name__ == "__main__":
    run_tactical_test()
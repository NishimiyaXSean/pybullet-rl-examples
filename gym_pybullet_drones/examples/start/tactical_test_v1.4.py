import time
import sys
import math
import numpy as np
import pybullet as p
import pybullet_data
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

def create_obstacle(x_pos, y_pos, z_pos):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # 门柱1
    col_id_1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.5, 1.0])
    vis_id_1 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.5, 1.0], rgbaColor=[0.6, 0.4, 0.2, 1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id_1, baseVisualShapeIndex=vis_id_1, basePosition=[x_pos, y_pos - 1.0, z_pos])
    
    # 门柱2
    col_id_2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.5, 1.0])
    vis_id_2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.5, 1.0], rgbaColor=[0.6, 0.4, 0.2, 1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id_2, baseVisualShapeIndex=vis_id_2, basePosition=[x_pos, y_pos + 1.0, z_pos])

    # 门梁
    col_id_3 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 1.5, 0.1])
    vis_id_3 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 1.5, 0.1], rgbaColor=[0.5, 0.3, 0.1, 1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id_3, baseVisualShapeIndex=vis_id_3, basePosition=[x_pos, y_pos, z_pos + 1.1])

def run_auto_flight():
    print("[INFO] 初始化自动轨迹追踪...")

    # 1. 设置初始位置 (在圆周的一个点上)
    RADIUS = 2.0
    INIT_XYZ = np.array([[RADIUS, 0.0, 1.0]]) 
    
    env = CtrlAviary(drone_model=DroneModel.CF2X,
                     num_drones=1,
                     initial_xyzs=INIT_XYZ,
                     physics=Physics.PYB,
                     pyb_freq=240,
                     ctrl_freq=240,
                     gui=True)
    
    # 【关键】一定要先 reset，再创建障碍物，否则障碍物会被重置掉
    obs, info = env.reset(seed=42)

    # 2. 放置障碍物 (放在圆周路径上的 x=0, y=2 的位置)
    create_obstacle(x_pos=0.0, y_pos=2.0, z_pos=1.0)
    create_obstacle(x_pos=0.0, y_pos=-2.0, z_pos=1.0)
    
    p.loadURDF("plane.urdf", [0, 0, 0])
    controller = DSLPIDControl(drone_model=DroneModel.CF2X)

    # 3. 追踪参数 
    SMOOTH_FACTOR = 0.1 
    current_target_pos = INIT_XYZ[0].copy()
    target_yaw = 0.0

    CTRL_FREQ = 240
    TOTAL_STEPS = 10000
    START_TIME = time.time()
    action = np.zeros((1, 4))

    print("------------------------------------------------")
    print("[AUTO] 正在执行自动圆形绕飞任务...")
    print("------------------------------------------------")

    for i in range(TOTAL_STEPS):
        # --- 1. 自动轨迹生成 (替代键盘输入) ---
        t = i / CTRL_FREQ
        angle = 0.5 * t  # 旋转速度
        
        # 计算理想的圆形路径坐标
        ideal_x = RADIUS * math.cos(angle)
        ideal_y = RADIUS * math.sin(angle)
        ideal_z = 1.0 + 0.2 * math.sin(t) # 稍微带一点高度起伏更真实
        
        # 让实际控制目标平滑地趋近理想路径 (防止动作过猛导致翻车)
        current_target_pos[0] = current_target_pos[0] * (1 - SMOOTH_FACTOR) + ideal_x * SMOOTH_FACTOR
        current_target_pos[1] = current_target_pos[1] * (1 - SMOOTH_FACTOR) + ideal_y * SMOOTH_FACTOR
        current_target_pos[2] = current_target_pos[2] * (1 - SMOOTH_FACTOR) + ideal_z * SMOOTH_FACTOR
        
        # 自动计算 Yaw，让机头始终朝向前进方向
        target_yaw = angle + math.pi/2

        # --- 2. PID 计算 ---
        action[0], _, _ = controller.computeControlFromState(
            control_timestep=1/CTRL_FREQ,
            state=obs[0],
            target_pos=current_target_pos,
            target_vel=np.zeros(3),
            target_rpy=np.array([0, 0, target_yaw])
        )

        # --- 3. 执行 ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- 4. 智能运镜 ---
        if i % 5 == 0:
            p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=np.degrees(target_yaw) - 90,
                cameraPitch=-20,
                cameraTargetPosition=obs[0][0:3]
            )
    
        # --- 5. 状态打印 ---
        if i % 48 == 0:
            sys.stdout.write(f"\r[AUTO] Step:{i} | Pos:({obs[0][0]:.1f}, {obs[0][1]:.1f}) | Yaw:{np.degrees(target_yaw):.1f}")
            sys.stdout.flush()

        # 同步时间
        elapsed = time.time() - START_TIME
        if elapsed < (i + 1) / CTRL_FREQ:
            time.sleep((i + 1) / CTRL_FREQ - elapsed)

    env.close()

if __name__ == "__main__":
    run_auto_flight()
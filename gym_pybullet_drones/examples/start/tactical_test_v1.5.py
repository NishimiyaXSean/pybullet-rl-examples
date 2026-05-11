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
    col_id_1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.5, 1.0])
    vis_id_1 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.5, 1.0], rgbaColor=[0.6, 0.4, 0.2, 1])
    p.createMultiBody(0, col_id_1, vis_id_1, [x_pos, y_pos - 1.0, z_pos])
    
    col_id_2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.5, 1.0])
    vis_id_2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.5, 1.0], rgbaColor=[0.6, 0.4, 0.2, 1])
    p.createMultiBody(0, col_id_2, vis_id_2, [x_pos, y_pos + 1.0, z_pos])

    col_id_3 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 1.5, 0.1])
    vis_id_3 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 1.5, 0.1], rgbaColor=[0.5, 0.3, 0.1, 1])
    p.createMultiBody(0, col_id_3, vis_id_3, [x_pos, y_pos, z_pos + 1.1])

def run_camera_control_flight():
    # 1. 环境初始化
    RADIUS = 2.0
    INIT_XYZ = np.array([[RADIUS, 0.0, 1.0]]) 
    
    env = CtrlAviary(drone_model=DroneModel.CF2X,
                     num_drones=1,
                     initial_xyzs=INIT_XYZ,
                     physics=Physics.PYB,
                     gui=True)
    
    obs, info = env.reset()
    create_obstacle(x_pos=0.0, y_pos=2.0, z_pos=1.0)
    create_obstacle(x_pos=0.0, y_pos=-2.0, z_pos=1.0)
    p.loadURDF("plane.urdf", [0, 0, 0])
    
    controller = DSLPIDControl(drone_model=DroneModel.CF2X)

    # --- 摄像头管理变量 ---
    # 0: 自动跟随模式 (默认)
    # 1: 自由视角模式 (手动控制)
    camera_mode = 0 

    CTRL_FREQ = 240
    START_TIME = time.time()
    action = np.zeros((1, 4))

    print("------------------------------------------------")
    print("[VIEW] 视角控制说明：")
    print("   按数字键 [1]: 开启自动跟随 (默认)")
    print("   按数字键 [2]: 切换为自由视角 (此时可用 Ctrl+鼠标 控制)")
    print("------------------------------------------------")

    for i in range(10000):
        # --- 2. 自动轨迹生成 (圆形) ---
        t = i / CTRL_FREQ
        angle = 0.6 * t
        target_pos = np.array([RADIUS * math.cos(angle), RADIUS * math.sin(angle), 1.0])
        target_yaw = angle + math.pi/2

        # --- 3. PID 控制 ---
        action[0], _, _ = controller.computeControlFromState(
            control_timestep=1/CTRL_FREQ,
            state=obs[0],
            target_pos=target_pos,
            target_rpy=np.array([0, 0, target_yaw])
        )
        obs, reward, terminated, truncated, info = env.step(action)

        # --- 4. ★ 键盘模式切换与摄像头逻辑 ★ ---
        keys = p.getKeyboardEvents()
        
        # 模式切换逻辑
        if ord('1') in keys and keys[ord('1')] & p.KEY_WAS_TRIGGERED:
            camera_mode = 0
            print("\r[VIEW] 模式：自动跟随模式           ", end='')
        elif ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED:
            camera_mode = 1
            print("\r[VIEW] 模式：自由视角模式 (手动操作)", end='')

        # 摄像头更新逻辑
        if camera_mode == 0:
            # 获取当前用户已经在调试窗口中手动调整过的参数（如距离和角度）
            # 这样跟随的时候会保留你的视角偏好，而不是生硬地重置
            cam_info = p.getDebugVisualizerCamera()
            cur_yaw = cam_info[8]
            cur_pitch = cam_info[9]
            cur_dist = cam_info[10]
            
            # 将目标点锁定在无人机位置 (obs[0][0:3])
            p.resetDebugVisualizerCamera(
                cameraDistance=cur_dist,
                cameraYaw=cur_yaw,
                cameraPitch=cur_pitch,
                cameraTargetPosition=obs[0][0:3]
            )
        else:
            # 在自由模式 (camera_mode == 1) 下，我们【不执行】resetDebugVisualizerCamera
            # 这允许 PyBullet 内部的 GUI 线程接管摄像头控制
            pass
            
        
        # 状态打印
        if i % 48 == 0:
            sys.stdout.write(f"\r[AUTO] Step:{i} | Pos:({obs[0][0]:.1f}, {obs[0][1]:.1f}) | Yaw:{np.degrees(target_yaw):.1f}")
            sys.stdout.flush()

        # --- 5. 同步时间 ---
        elapsed = time.time() - START_TIME
        if elapsed < (i + 1) / CTRL_FREQ:
            time.sleep((i + 1) / CTRL_FREQ - elapsed)

    env.close()

if __name__ == "__main__":
    run_camera_control_flight()
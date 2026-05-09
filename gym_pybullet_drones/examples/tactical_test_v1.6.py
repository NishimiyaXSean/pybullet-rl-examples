import time
import sys
import math
import numpy as np
import pybullet as p
import pybullet_data
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

def create_moving_obstacle(x_pos, y_pos, z_pos):
    """ 创建并返回障碍物组件的 ID，以便后续移动 """
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 门柱尺寸
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.2, 1.0])
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.2, 1.0], rgbaColor=[0.8, 0.1, 0.1, 1]) # 红色警示
    
    # 创建左右两个柱子
    id_l = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=[x_pos, y_pos - 0.6, z_pos])
    id_r = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=[x_pos, y_pos + 0.6, z_pos])
    
    # 门梁
    col_b = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.8, 0.1])
    vis_b = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.8, 0.1], rgbaColor=[0.7, 0.1, 0.1, 1])
    id_b = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_b, baseVisualShapeIndex=vis_b, basePosition=[x_pos, y_pos, z_pos + 1.1])
    
    return [id_l, id_r, id_b]

def run_dynamic_flight():
    # 1. 初始化
    RADIUS = 2.5
    INIT_XYZ = np.array([[RADIUS, 0.0, 1.0]]) 
    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=INIT_XYZ, gui=True)
    obs, info = env.reset()
    
    # 2. 创建动态障碍物并获取其 ID
    # 把它放在圆周路径的一侧 (x=0, y=2.5)
    gate_ids = create_moving_obstacle(x_pos=0.0, y_pos=RADIUS, z_pos=1.0)
    
    p.loadURDF("plane.urdf", [0, 0, 0])
    controller = DSLPIDControl(drone_model=DroneModel.CF2X)

    camera_mode = 0 
    CTRL_FREQ = 240
    START_TIME = time.time()
    
    print("[FLY] 动态障碍物挑战：计算好时机穿过移动的红门！")

    for i in range(15000):
        t = i / CTRL_FREQ
        
        # --- 3. 动态障碍物逻辑 (核心) ---
        # 让门在 Y 轴方向以中心点为准，上下摆动 1.2 米
        # 摆动周期由 1.5 * t 决定
        y_offset = 2 * math.sin(1.5 * t)  #修改系数，改变动态障碍物移动速度/周期
        base_y = RADIUS
        
        # 更新三个组件的位置
        p.resetBasePositionAndOrientation(gate_ids[0], [0.0, base_y + y_offset - 0.6, 1.0], [0,0,0,1])
        p.resetBasePositionAndOrientation(gate_ids[1], [0.0, base_y + y_offset + 0.6, 1.0], [0,0,0,1])
        p.resetBasePositionAndOrientation(gate_ids[2], [0.0, base_y + y_offset, 2.1], [0,0,0,1])

        # --- 4. 无人机轨迹 (依然是圆周) ---
        angle = 0.2 * t  #修改系数，改变无人机基础速度
        target_pos = np.array([RADIUS * math.cos(angle), RADIUS * math.sin(angle), 1.0])
        target_yaw = angle + math.pi/2

        action, _, _ = controller.computeControlFromState(
            control_timestep=1/CTRL_FREQ,
            state=obs[0],
            target_pos=target_pos,
            target_rpy=np.array([0, 0, target_yaw])
        )
        obs, reward, terminated, truncated, info = env.step(action.reshape(1, 4))

        # --- 5. 摄像头切换逻辑 (继承上次代码) ---
        keys = p.getKeyboardEvents()
        if ord('1') in keys and keys[ord('1')] & p.KEY_WAS_TRIGGERED: camera_mode = 0
        if ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED: camera_mode = 1

        if camera_mode == 0:
            cam_info = p.getDebugVisualizerCamera()
            p.resetDebugVisualizerCamera(cameraDistance=cam_info[10], cameraYaw=cam_info[8], 
                                         cameraPitch=cam_info[9], cameraTargetPosition=obs[0][0:3])

        # --- 6. 简单的碰撞处理 ---
        # 如果无人机倾斜过大，说明撞到了
        if abs(obs[0][7]) > 0.8 or abs(obs[0][8]) > 0.8:
            print("\n[BOOM] 你被移动门撞飞了！")
            break

        # 同步
        elapsed = time.time() - START_TIME
        if elapsed < t:
            time.sleep(t - elapsed)

    env.close()

if __name__ == "__main__":
    run_dynamic_flight()


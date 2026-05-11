import time
import math
import numpy as np
import pybullet as p
import cv2  # 用于显示摄像头画面
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics

def create_environment_objects():
    """ 在场景里乱放一些彩色障碍物，方便视觉观察 """
    # 放几个红色的方块
    for i in range(5):
        pos = [np.random.uniform(-3, 3), np.random.uniform(-3, 3), 0.5]
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.5])
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.5], rgbaColor=[1, 0, 0, 1])
        p.createMultiBody(0, col_id, vis_id, basePosition=pos)

def get_fpv_image(drone_id, width=320, height=240):
    """
    获取无人机的第一人称视角画面
    """
    # 1. 获取无人机的位置和姿态
    pos, quat = p.getBasePositionAndOrientation(drone_id)
    rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    
    # 2. 计算摄像头观察点
    # 假设摄像头在机头前方 0.1米处
    forward_vec = rot_mat @ np.array([1, 0, 0])  # 机体坐标系的X轴是正前方
    up_vec = rot_mat @ np.array([0, 0, 1])       # 机体坐标系的Z轴是上方
    camera_pos = pos + rot_mat @ np.array([0.1, 0, 0]) # 摄像头稍微前移
    target_pos = camera_pos + forward_vec        # 镜头盯着前方看
    
    # 3. 构建视图矩阵
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_pos,
        cameraTargetPosition=target_pos,
        cameraUpVector=up_vec
    )
    
    # 4. 构建投影矩阵 (FOV=60度, 近平面0.1, 远平面100)
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=width/height,
        nearVal=0.1,
        farVal=100.0
    )
    
    # 5. 渲染图像
    # renderer=p.ER_TINY_RENDERER 是 CPU 渲染，如果是 NVIDIA 显卡可以用 p.ER_BULLET_HARDWARE_OPENGL
    (w, h, rgb, depth, seg) = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER 
    )
    
    # 6. 处理返回的图像数据
    rgb_array = np.reshape(rgb, (h, w, 4)).astype(np.uint8)
    rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR) # 转为 OpenCV 格式
    
    return rgb_array

def run_vision_flight():
    # 初始化环境
    RADIUS = 2
    INIT_XYZ = np.array([[RADIUS, 0.0, 0.2]]) 
    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1,initial_xyzs=INIT_XYZ, gui=True)
    obs, info = env.reset()
    
    create_environment_objects()
    controller = DSLPIDControl(drone_model=DroneModel.CF2X)
    
    CTRL_FREQ = 240
    START_TIME = time.time()
    
    print("[VISION] 摄像头已启动！请查看弹出窗口...")

    for i in range(20000):
        t = i / CTRL_FREQ
        
        # --- 1. 轨迹控制 (绕圆飞) ---
        angle = 0.5 * t
        target_pos = np.array([RADIUS * math.cos(angle), RADIUS * math.sin(angle), 0.2])
        target_yaw = angle + math.pi/2
        
        action, _, _ = controller.computeControlFromState(
            control_timestep=1/CTRL_FREQ,
            state=obs[0],
            target_pos=target_pos,
            target_rpy=np.array([0, 0, target_yaw])
        )
        obs, reward, terminated, truncated, info = env.step(action.reshape(1, 4))

        # --- 2. 视觉渲染 (每 4 帧渲染一次，节省 CPU) ---
        if i % 4 == 0:
            frame = get_fpv_image(env.DRONE_IDS[0])
            
            # 在图像上加个简单的准星
            cv2.line(frame, (150, 120), (170, 120), (0, 255, 0), 1)
            cv2.line(frame, (160, 110), (160, 130), (0, 255, 0), 1)
            
            cv2.imshow('Drone FPV', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 同步
        elapsed = time.time() - START_TIME
        if elapsed < t:
            time.sleep(t - elapsed)

    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    run_vision_flight()
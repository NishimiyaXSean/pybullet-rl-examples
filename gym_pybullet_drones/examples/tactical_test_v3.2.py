import time
import math
import numpy as np
import pybullet as p
import cv2
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics

def create_obstacle_wall(pos):
    """ 在路径上创建一个宽大的障碍墙，强迫无人机避障 """
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 1.2, 1.0])
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 1.2, 1.0], rgbaColor=[0.2, 0.2, 0.8, 1])
    p.createMultiBody(0, col_id, vis_id, basePosition=pos)

def get_vision_data(drone_id, width=160, height=120):
    """ 获取RGB和深度图 """
    pos, quat = p.getBasePositionAndOrientation(drone_id)
    rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    
    forward_vec = rot_mat @ np.array([1, 0, 0])
    up_vec = rot_mat @ np.array([0, 0, 1])
    camera_pos = pos + rot_mat @ np.array([0.1, 0, 0])
    
    view_matrix = p.computeViewMatrix(camera_pos, camera_pos + forward_vec, up_vec)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60.0, aspect=width/height, nearVal=0.1, farVal=10.0)
    
    # 渲染
    _, _, rgb, depth, _ = p.getCameraImage(width, height, view_matrix, proj_matrix, renderer=p.ER_TINY_RENDERER)
    
    # 处理RGB
    rgb_frame = np.reshape(rgb, (height, width, 4)).astype(np.uint8)
    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGBA2BGR)
    
    # 处理深度图 (将其转换为 0-255 灰度图用于显示，并保留物理距离用于逻辑)
    # PyBullet的深度是0-1之间的非线性值，这里简化处理用于演示
    depth_frame = np.reshape(depth, (height, width))
    # 转换为实际距离 (米) - 这是一个近似公式
    near, far = 0.1, 10.0
    actual_depth = far * near / (far - (far - near) * depth_frame)
    
    return rgb_frame, actual_depth

def run_vision_avoidance():
    RADIUS = 3.0
    INIT_XYZ = np.array([[RADIUS, 0.0, 0.2]]) 
    env = CtrlAviary(drone_model=DroneModel.CF2X,num_drones=1,initial_xyzs=INIT_XYZ , gui=True)
    obs, info = env.reset()
    
    # 放置墙壁 (增加到两面墙)
    create_obstacle_wall([RADIUS * math.cos(math.pi/4), RADIUS * math.sin(math.pi/4), 1.0])
    create_obstacle_wall([RADIUS * math.cos(math.pi), RADIUS * math.sin(math.pi), 1.0])

    controller = DSLPIDControl(drone_model=DroneModel.CF2X)
    CTRL_FREQ = 240
    START_TIME = time.time()
    current_angle = 0.0
    
    # 避障偏移量
    side_offset = 0.0 
    
    print("[SYSTEM] 深度视觉避障系统启动...")

    for i in range(20000):
        # --- 1. 获取视觉数据 ---
        rgb, depth = get_vision_data(env.DRONE_IDS[0])
        
        # --- 2. 深度视觉逻辑 ---
        h, w = depth.shape

        # 截取中间关键感应区（去掉画面最边缘，防止无人机因为看到地面而误判）
        roi_depth = depth[h//4 : 3*h//4, :] 

        # 将画面垂直切分为左、中、右三块
        left_zone = depth[:, :w//3]
        center_zone = depth[:, w//3:2*w//3]
        right_zone = depth[:, 2*w//3:]
        
        # 使用 min() 获取最近物体的距离，而不是平均值
        min_dist_center = np.min(center_zone)
        min_dist_left = np.min(left_zone)
        min_dist_right = np.min(right_zone)

        
        # 避障决策
        speed = 0.3 # 默认速度
        if min_dist_center < 2.5: # 探测距离拉长到 2.5米
            # 情况 A: 前方有墙，且很近了
            speed = 0.01 # 大幅减速，给自己留反应时间
            
            # 决定往哪偏：对比左右哪边更空旷
            if min_dist_left > min_dist_right:
                side_offset += 0.01 # 强力向左修正
            else:
                side_offset -= 0.01 # 强力向右修正
                
            print(f"\r[AVOID] 距离墙体: {min_dist_center:.2f}m | 偏移量: {side_offset:.2f}  ", end="")
        else:
            # 情况 B: 前方开阔，尝试平滑回到轨道
            side_offset *= 0.98 
            
        # 限制最大偏移量，防止无人机飞出宇宙
        side_offset = np.clip(side_offset, -1.5, 1.5)

        # --- 3. 结合避障修正轨迹 ---
        dt = 1/CTRL_FREQ
        current_angle += speed * dt
        
        # 原始圆心坐标
        base_x = RADIUS * math.cos(current_angle)
        base_y = RADIUS * math.sin(current_angle)
        
        # 计算切线方向和法线方向（用于施加偏移）
        # side_offset 作用于法线方向（半径方向）
        final_radius = RADIUS + side_offset
        target_pos = np.array([
            final_radius * math.cos(current_angle), 
            final_radius * math.sin(current_angle), 
            1.2
        ])
        target_yaw = current_angle + math.pi/2

        # --- 4. 控制 ---
        action, _, _ = controller.computeControlFromState(dt, obs[0], target_pos, target_rpy=np.array([0,0,target_yaw]))
        obs, _, _, _, _ = env.step(action.reshape(1, 4))

        # --- 5. 视觉效果展示 ---
        if i % 6 == 0:
            # 将深度图转为彩色，红色表示危险
            depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            
            # 在图像上画出探测到的最小距离（调试用）
            cv2.putText(depth_color, f"Center Min: {min_dist_center:.1f}m", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.imshow('RGB FPV', rgb)
            cv2.imshow('Safety Radar (Depth)', depth_color)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        # 同步
        time.sleep(max(0, dt - (time.time() - (START_TIME if i==0 else last_step_time))))
        last_step_time = time.time()    

    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    run_vision_avoidance()
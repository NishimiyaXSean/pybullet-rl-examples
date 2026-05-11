import time
import math
import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics

def create_pillar(pos):
    """ 
    使用 GEOM_BOX 创建方形柱子，避开 GEOM_CYLINDER 变态的参数名问题
    halfExtents=[宽/2, 厚/2, 高/2]
    """
    half_extents = [0.2, 0.2, 1.0] # 宽0.4, 深0.4, 高2.0 的柱子
    
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.1, 0.8, 0.1, 1])
    
    # 柱子中心点在 pos，因为高度是 2.0，所以中心点在 1.0 处正好立在地面上
    p.createMultiBody(baseMass=0, 
                      baseCollisionShapeIndex=col_id, 
                      baseVisualShapeIndex=vis_id, 
                      basePosition=pos)

def get_lidar_data(drone_id, num_rays, ray_length):
    """
    模拟激光雷达传感器
    返回: 一个布尔列表，表示对应的射线是否检测到物体
    """
    # 获取无人机位置和姿态
    pos, quat = p.getBasePositionAndOrientation(drone_id)
    rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    
    ray_starts = []
    ray_ends = []
    
    # 定义射线的角度（在无人机前方扇形展开）
    angles = np.linspace(-math.pi/3, math.pi/3, num_rays) # 正前方 120度 扇形
    
    for angle in angles:
        # 在机体坐标系下的射线终点
        # x轴朝前，y轴朝左
        local_ray_end = [ray_length * math.cos(angle), ray_length * math.sin(angle), 0]
        # 转换到世界坐标系
        world_ray_end = pos + rot_mat @ local_ray_end
        
        ray_starts.append(pos)
        ray_ends.append(world_ray_end)
    
    # 执行批量射线检测
    results = p.rayTestBatch(ray_starts, ray_ends)
    
    # 处理结果
    hit_flags = []
    for i, res in enumerate(results):
        hit_object_id = res[0]
        hit_fraction = res[2] # 0.0 到 1.0 之间的比例
        
        # 如果撞到了物体且不是自己
        if hit_object_id != -1 and hit_object_id != drone_id:
            hit_flags.append(True)
            # 可视化：检测到物体变红
            p.addUserDebugLine(ray_starts[i], ray_ends[i], [1, 0, 0], lineWidth=2, lifeTime=0.1)
        else:
            hit_flags.append(False)
            # 可视化：未检测到物体变绿
            p.addUserDebugLine(ray_starts[i], ray_ends[i], [0, 1, 0], lineWidth=1, lifeTime=0.1)
            
    return hit_flags

def run_avoidance_flight():
    RADIUS = 3
    INIT_XYZ = np.array([[RADIUS, 0.0, 1.0]]) 
    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=INIT_XYZ, gui=True)
    obs, info = env.reset()
    
    # 在圆周路径的 45度、135度和 225度 位置各放一个
    for deg in [45, 135, 225]:
        rad = math.radians(deg)
        create_pillar([RADIUS * math.cos(rad), RADIUS * math.sin(rad), 1.0])

    controller = DSLPIDControl(drone_model=DroneModel.CF2X)
    CTRL_FREQ = 240
    START_TIME = time.time()

    # 轨迹管理变量
    current_angle = 0.0  # 使用累加角度，防止速度变化时发生跳变
    
    print("------------------------------------------------")
    print("[SYSTEM] 避障演示已启动！")
    print("   - 绿色射线：路径安全")
    print("   - 红色射线：发现障碍物，触发紧急减速")
    print("------------------------------------------------")

    for i in range(20000):
        
        # --- 1. 获取传感器数据 ---
        # 探测前方 num_rays 个方向，长度 ray_length 米
        lidar_hits = get_lidar_data(env.DRONE_IDS[0], num_rays=7, ray_length=1.5)
        
        # --- 2. 避障决策逻辑 (简单的反应式控制) ---
        is_blocked = any(lidar_hits) # 只要任何一根射线照到物体
        
        if is_blocked:
            # 发现障碍物：减速并尝试偏离
            speed = -0.3  
            print(f"\r[WARNING] 检测到障碍物！紧急减速...", end="")
        else:
            speed = 0.6 # 正常飞行速度
            
        # --- 3. 轨迹生成 ---
        dt = 1/CTRL_FREQ
        current_angle += speed * dt
        
        target_pos = np.array([RADIUS * math.cos(current_angle), RADIUS * math.sin(current_angle), 1.0])
        target_yaw = current_angle + math.pi/2

        action, _, _ = controller.computeControlFromState(
            control_timestep=1/CTRL_FREQ,
            state=obs[0],
            target_pos=target_pos,
            target_rpy=np.array([0, 0, target_yaw])
        )
        obs, reward, terminated, truncated, info = env.step(action.reshape(1, 4))

        # 实时同步
        elapsed = time.time() - START_TIME
        if elapsed < i * dt:
            time.sleep(i * dt - elapsed)

    env.close()

if __name__ == "__main__":
    run_avoidance_flight()
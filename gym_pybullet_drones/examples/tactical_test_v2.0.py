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
    """
    使用 PyBullet 底层 API 创建一个简单的门框障碍物
    展示如何混合使用 gym-drones 和原生 PyBullet
    """
    # 获取默认纹理路径
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 简单拼凑3个长方体组成门
    # 门柱1
    col_id_1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.5, 1.0]) # 长宽高的一半
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

def run_manual_flight():
    print("[INFO] 初始化飞行模拟器...")

    # 1. 单机环境
    INIT_XYZ = np.array([[0.0, 0.0, 1.0]]) # 起飞点


    # 定义平滑系数 (0.0 ~ 1.0)
    # 值越小，手感越"肉"（惯性大，平滑）；值越大，反应越灵敏（更生硬）
    SMOOTH_FACTOR = 0.1 

    # 记录两个位置变量
    current_target_pos = INIT_XYZ[0].copy() # 实际发送给控制器的位置
    user_input_pos = INIT_XYZ[0].copy()     # 键盘控制的“虚拟”位置
    
    env = CtrlAviary(drone_model=DroneModel.CF2X,
                     num_drones=1,
                     initial_xyzs=INIT_XYZ,
                     physics=Physics.PYB,
                     pyb_freq=240,
                     ctrl_freq=240,
                     gui=True,
                     record=False)
    
    obs, info = env.reset(seed=42)

    # 2. 创造障碍物 (在 x=2.0 处放置一个门)
    create_obstacle(x_pos=3.0, y_pos=0.0, z_pos=1.0)
    
    # 地面加个纹理,更有速度感
    p.loadURDF("plane.urdf", [0, 0, 0])

    controller = DSLPIDControl(drone_model=DroneModel.CF2X)
    

    # 3. 键盘控制参数
    target_pos = INIT_XYZ[0].copy() # 初始目标就是当前位置
    target_yaw = 0.0
    

    # 4. 摄像头模式 (追尾视角)
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-90, cameraPitch=-20, cameraTargetPosition=target_pos)

    print("------------------------------------------------")
    print("[CONTROL] 键盘操作说明：")
    print("   ↑ / ↓ : 前进 / 后退 (X轴)")
    print("   ← / → : 左移 / 右移 (Y轴)")
    print("   W / S : 上升 / 下降 (Z轴)")
    print("   A / D : 左转 / 右转 (Yaw)")
    print("[GOAL] 尝试穿过前方的门框！")
    print("------------------------------------------------")
    input("按回车键开始驾驶 >> ")

    CTRL_FREQ = 240
    TOTAL_STEPS = 20000 # 玩久一点
    START_TIME = time.time()
    action = np.zeros((1, 4))

    # 速度灵敏度
    Z_SPEED = 0.01
    STEP_SIZE = 0.005
    YAW_STEP = 0.015

    for i in range(TOTAL_STEPS):
        
        # --- 1. 读取键盘输入 ---
        keys = p.getKeyboardEvents()
        
        # 定义机体坐标系下的输入向量 (local_x: 前后, local_y: 左右)
        local_x = 0.0
        local_y = 0.0
        val_z = 0.0
        yaw_input = 0.0

        # 读取按键 (这里假设: 上下=前后, 左右=左右平移, q/e=旋转)
        # 注意：你需要根据你的习惯绑定按键
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            local_x = STEP_SIZE     # 向前
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            local_x = -STEP_SIZE    # 向后
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            local_y = STEP_SIZE     # 向左
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            local_y = -STEP_SIZE    # 向右
            
        # 高度控制
        if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
            val_z = Z_SPEED
        if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
            val_z = -Z_SPEED
            
        # 旋转控制 (Yaw)
        if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
            yaw_input = YAW_STEP    # 左转
        if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
            yaw_input = -YAW_STEP   # 右转
        
        # 获取无人机当前的真实姿态 (主要是 Yaw)
        # env.DRONE_IDS[0] 是第一架无人机的 PyBullet ID
        pos, quat = p.getBasePositionAndOrientation(env.DRONE_IDS[0])
        rpy = p.getEulerFromQuaternion(quat)
        current_yaw = rpy[2] # [Roll, Pitch, Yaw]

        # 将机体坐标系的输入 (local_x, local_y) 旋转到世界坐标系
        # 使用旋转公式
        world_dx = local_x * math.cos(current_yaw) - local_y * math.sin(current_yaw)
        world_dy = local_x * math.sin(current_yaw) + local_y * math.cos(current_yaw)

        # 更新用户意图位置
        user_input_pos[0] += world_dx
        user_input_pos[1] += world_dy
        user_input_pos[2] += val_z
        target_yaw += yaw_input * 0.5

        # 让实际目标位置平滑地趋近于用户意图位置
        # 公式：新位置 = 旧位置 * (1 - alpha) + 目标位置 * alpha
        current_target_pos[0] = current_target_pos[0] * (1 - SMOOTH_FACTOR) + user_input_pos[0] * SMOOTH_FACTOR
        current_target_pos[1] = current_target_pos[1] * (1 - SMOOTH_FACTOR) + user_input_pos[1] * SMOOTH_FACTOR
        current_target_pos[2] = current_target_pos[2] * (1 - SMOOTH_FACTOR) + user_input_pos[2] * SMOOTH_FACTOR
       
        # 限制高度不能钻入地下
        if target_pos[2] < 0.1: 
            target_pos[2] = 0.1

        # --- 2. PID 计算 ---
        # 注意：这里我们不断改变 target_pos，PID 会努力把飞机拉过去
        # 这就是最基础的 "设定点控制 (Setpoint Control)"
        action[0], _, _ = controller.computeControlFromState(
            control_timestep=1/CTRL_FREQ,
            state=obs[0],
            target_pos=current_target_pos,
            target_vel=np.zeros(3),
            target_rpy=np.array([0, 0, target_yaw]) # 传入期望的 Yaw
        )

        # --- 3. 执行 ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- 4. 智能运镜: 追尾模式 ---
        # 摄像头紧跟在飞机屁股后面，带一点平滑
        current_pos = obs[0][0:3]
        # 简单算法：摄像头在飞机后方 1.5米，高度高 0.5米
        # 为了更真实的追尾，我们需要根据飞机的 Yaw 来计算摄像头的偏移
        # 这里为了简化，我们仅使用位置跟随
        p.resetDebugVisualizerCamera(
            cameraDistance=1,
            cameraYaw=np.degrees(target_yaw) - 90, # 随飞机转向
            cameraPitch=-20,
            cameraTargetPosition=current_pos
        )

        # --- 5. 碰撞检测 ---
        # 如果飞机位置极其接近障碍物坐标，或者翻车了
        rpy = obs[0][7:10] # 获取 Roll, Pitch, Yaw
        if abs(rpy[0]) > 1.0 or abs(rpy[1]) > 1.0: # 倾角超过 1弧度 (约57度) 判定为撞击失控
            print(f"\n[CRASH] 发生碰撞！ 游戏结束。 步数: {i}")
            break

        # 打印简报
        if i % 24 == 0:
            sys.stdout.write(f"\r[PILOT] Pos:({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f}) | Yaw:{np.degrees(target_yaw):.1f}   ")
            sys.stdout.flush()

        # 同步
        elapsed = time.time() - START_TIME
        if elapsed < (i + 1) / CTRL_FREQ:
            time.sleep((i + 1) / CTRL_FREQ - elapsed)

    print("\n[INFO] 模拟结束。")
    time.sleep(2)
    env.close()

if __name__ == "__main__":
    run_manual_flight()
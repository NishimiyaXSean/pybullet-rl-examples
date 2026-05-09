import time
import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

def run_camera_tracking_test():
    print("[INFO] 初始化智能运镜测试...")

    # --- 1. 初始位置 ---
    INIT_XYZS = np.array([
        [0.0, 0.0, 1.0],    # 领机
        [-0.5, -0.5, 1.0]   # 僚机
    ])
    INIT_RPYS = np.array([[0,0,0], [0,0,0]])

    # --- 2. 环境配置 ---
    env = CtrlAviary(drone_model=DroneModel.CF2X,
                     num_drones=2,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=Physics.PYB,
                     pyb_freq=240,
                     ctrl_freq=240,
                     gui=True,
                     record=False)

    controllers = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
    obs, info = env.reset(seed=42)

    # --- 3. 摄像头管理变量 ---
    # 0: 编队中心跟随 (默认)
    # 1: 领机尾随
    # 2: 全局静态上帝视角
    camera_mode = 0 
    
    # 初始化摄像头参数
    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

    print("------------------------------------------------")
    print("[CONTROL] 键盘快捷键说明：")
    print("   [1] 切换到：编队中心跟随 (推荐)")
    print("   [2] 切换到：领机近距离特写")
    print("   [3] 切换到：固定上帝视角")
    print("[ACTION] 按回车键起飞...")
    print("------------------------------------------------")
    input()

    CTRL_FREQ = 240
    TOTAL_STEPS = 10000 
    START_TIME = time.time()
    action = np.zeros((2, 4))
    
    # 轨迹记录
    last_pos_0 = obs[0][0:3]
    last_pos_1 = obs[1][0:3]

    for i in range(TOTAL_STEPS):
        current_time = i / CTRL_FREQ
        
        # --- 战术动作 (8字 + 跟随) ---
        t = current_time * 0.5
        target_pos_0 = np.array([1.5 * np.sin(t), 1.5 * np.sin(t) * np.cos(t), 1.0])
        target_pos_1 = obs[0][0:3] + np.array([-0.4, -0.4, 0.4]) # 僚机跟随

        # PID 计算
        action[0], _, _ = controllers[0].computeControlFromState(1/CTRL_FREQ, obs[0], target_pos_0, np.zeros(3))
        action[1], _, _ = controllers[1].computeControlFromState(1/CTRL_FREQ, obs[1], target_pos_1, np.zeros(3))

        # 执行
        obs, reward, terminated, truncated, info = env.step(action)

        # --- 画轨迹 ---
        if i % 10 == 0:
            p.addUserDebugLine(last_pos_0, obs[0][0:3], [1, 0, 0], 10, 2)
            p.addUserDebugLine(last_pos_1, obs[1][0:3], [0, 1, 0], 10, 2)
            last_pos_0 = obs[0][0:3]
            last_pos_1 = obs[1][0:3]

        # =========================================================
        # ★★★ 核心部分：智能摄像头控制 ★★★
        # =========================================================
        
        # 1. 监听键盘输入 (PyBullet 内置函数)
        # ord('1') 获取字符的ASCII码
        keys = p.getKeyboardEvents()
        if ord('1') in keys and keys[ord('1')] & p.KEY_WAS_TRIGGERED:
            camera_mode = 0
            print("\r[VIEW] 切换视角: 编队中心跟随           ", end='')
        elif ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED:
            camera_mode = 1
            print("\r[VIEW] 切换视角: 领机特写               ", end='')
        elif ord('3') in keys and keys[ord('3')] & p.KEY_WAS_TRIGGERED:
            camera_mode = 2
            print("\r[VIEW] 切换视角: 固定上帝视角           ", end='')

        # 2. 获取当前摄像头状态 (保留用户的 Yaw/Pitch 旋转操作)
        # getDebugVisualizerCamera 返回一个元组，第10个是yaw, 8是pitch, 10是dist
        cam_info = p.getDebugVisualizerCamera()
        current_yaw = cam_info[8]
        current_pitch = cam_info[9]
        current_dist = cam_info[10]

        # 3. 根据模式计算新的聚焦点 (Target Position)
        new_target = [0, 0, 0]
        update_cam = False

        if camera_mode == 0: # 编队中心
            pos0 = obs[0][0:3]
            pos1 = obs[1][0:3]
            # 计算两机中心点
            center_pos = (pos0 + pos1) / 2
            new_target = center_pos
            update_cam = True
            
        elif camera_mode == 1: # 领机特写
            new_target = obs[0][0:3]
            update_cam = True
            # 特写模式拉近一点距离 (如果用户没手动拉远的话，可以强制设个值，这里选择保留用户设置)
            # current_dist = 1.5 
            
        elif camera_mode == 2: # 固定视角
            # 不更新摄像头，保持静止
            update_cam = False

        # 4. 应用更新
        if update_cam:
            p.resetDebugVisualizerCamera(cameraDistance=current_dist,
                                         cameraYaw=current_yaw,
                                         cameraPitch=current_pitch,
                                         cameraTargetPosition=new_target)

        # =========================================================

        # 同步时间
        elapsed = time.time() - START_TIME
        if elapsed < (i + 1) / CTRL_FREQ:
            time.sleep((i + 1) / CTRL_FREQ - elapsed)

    env.close()

if __name__ == "__main__":
    run_camera_tracking_test()
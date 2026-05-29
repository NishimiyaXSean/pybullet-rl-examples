import math

def generate_minimal_acmi():
    filename = "demo_dogfight.txt.acmi"
    
    with open(filename, "w", encoding="utf-8") as f:
        # ==========================================
        # 1. 写入文件头 (Header)
        # ==========================================
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.1\n")
        
        # 设定全局属性 (时间基准和世界原点的经纬度)
        # 时间设为当前的 2026-05-23
        f.write("0,ReferenceTime=2026-05-23T10:14:19Z\n")
        # 假设我们将 PyBullet 的 (0,0,0) 原点映射到地球某处，比如太平洋上空
        f.write("0,ReferenceLongitude=130.0\n")
        f.write("0,ReferenceLatitude=20.0\n")

        # ==========================================
        # 2. 注册物体并赋初值 (Object Registration)
        # ==========================================
        # 格式: ID,T=经度|纬度|高度|滚转|俯仰|偏航,Type=类型,Name=名称,Color=颜色
        # 101 代表攻击机，102 代表目标机
        f.write("101,T=130.0|20.0|3000|0|0|0,Type=Air+FixedWing,Name=Attacker,Color=Red\n")
        f.write("102,T=130.01|20.01|3000|0|0|0,Type=Air+FixedWing,Name=Evader,Color=Blue\n")

        # ==========================================
        # 3. 生成随时间变化的轨迹流 (Time Frames)
        # ==========================================
        # 模拟 60 帧，假设频率为 10Hz (每帧 0.1 秒)
        for step in range(60):
            time_sec = step * 0.1
            
            # 写入时间戳 (必须以 # 开头)
            f.write(f"#{time_sec:.2f}\n")
            
            # 模拟攻击机 (101) 向东飞行并右转
            # 注意 Tacview 的姿态顺序是 Roll(滚转)|Pitch(俯仰)|Yaw(偏航)，单位是【度】
            lon_101 = 130.0 + (step * 0.0001)
            yaw_101 = step * 1.5  
            f.write(f"101,T={lon_101:.6f}|20.0|3000|0|0|{yaw_101:.1f}\n")
            
            # 模拟目标机 (102) 保持盘旋 (结合你在环境里写的 evader_maneuver)
            lat_102 = 20.01 + math.sin(step * 0.1) * 0.001
            lon_102 = 130.01 + math.cos(step * 0.1) * 0.001
            roll_102 = 60.0 # 60度大坡度盘旋
            f.write(f"102,T={lon_102:.6f}|{lat_102:.6f}|3000|{roll_102}|0|0\n")

    print(f"✅ 成功生成 Tacview 日志: {filename}")

if __name__ == "__main__":
    generate_minimal_acmi()
import math
import os
from datetime import datetime

class TacviewLogger:
    def __init__(self, filename="combat_log.txt.acmi", base_lat=20.0, base_lon=130.0):
        """
        初始化 Tacview 记录器
        :param filename: 输出文件名 (需以 .txt.acmi 结尾)
        :param base_lat: 虚拟战场的基准纬度 (默认 20.0)
        :param base_lon: 虚拟战场的基准经度 (默认 130.0)
        """
        self.filename = filename
        self.base_lat = base_lat
        self.base_lon = base_lon
        
        # 经纬度估算常数 (赤道附近 1 度约等于 111.32 km)
        self.METERS_PER_DEG_LAT = 111320.0
        self.METERS_PER_DEG_LON = self.METERS_PER_DEG_LAT * math.cos(math.radians(base_lat))

        self.file = None
        self.is_recording = False
        self.start_time = 0.0

    def start(self, attacker_name="Attacker_0", evader_name="Evader_0"):
        """打开文件并写入 ACMI 头部和对象注册信息"""
        self.file = open(self.filename, "w", encoding="utf-8")
        self.is_recording = True
        
        # 1. 写入 Header
        time_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        self.file.write("FileType=text/acmi/tacview\n")
        self.file.write("FileVersion=2.1\n")
        self.file.write(f"0,ReferenceTime={time_str}\n")
        self.file.write(f"0,ReferenceLongitude={self.base_lon}\n")
        self.file.write(f"0,ReferenceLatitude={self.base_lat}\n")

        # 2. 注册无人机对象 (101: 主机, 102: 目标机)
        # Type 设置为 Drone 或 Air+FixedWing 均可
        self.file.write(f"101,T={self.base_lon}|{self.base_lat}|0|0|0|0,Type=Air+FixedWing,Name={attacker_name},Color=Red\n")
        self.file.write(f"102,T={self.base_lon}|{self.base_lat}|0|0|0|0,Type=Air+FixedWing,Name={evader_name},Color=Blue\n")

    def _convert_coords(self, x, y, z):
        """将 PyBullet 的 XYZ (米) 转换为经度、纬度、高度"""
        lat = self.base_lat + (y / self.METERS_PER_DEG_LAT)
        lon = self.base_lon + (x / self.METERS_PER_DEG_LON)
        return lon, lat, z

    def log_frame(self, time_sec, attacker_state, evader_state):
        """
        记录单帧数据
        :param state: 包含 pos (3维) 和 rpy (3维) 的字典或列表
        """
        if not self.is_recording or self.file is None:
            return

        # 写入时间戳
        self.file.write(f"#{time_sec:.3f}\n")

        # 解析并写入攻击机 (101) 状态
        a_pos, a_rpy = attacker_state['pos'], attacker_state['rpy']
        a_lon, a_lat, a_z = self._convert_coords(*a_pos)
        # PyBullet 的 rpy 是 (Roll, Pitch, Yaw) 弧度制，Tacview 需要角度制
        a_roll, a_pitch, a_yaw = [math.degrees(angle) for angle in a_rpy]

        # === 核心修复：坐标系对齐 ===
        # PyBullet (0=东, 逆时针) 转换为 Tacview (0=北, 顺时针)
        a_tacview_yaw = (90.0 - a_yaw) % 360.0

        self.file.write(f"101,T={a_lon:.7f}|{a_lat:.7f}|{a_z:.1f}|{a_roll:.1f}|{a_pitch:.1f}|{a_tacview_yaw:.1f}\n")

        # 解析并写入目标机 (102) 状态
        e_pos, e_rpy = evader_state['pos'], evader_state['rpy']
        e_lon, e_lat, e_z = self._convert_coords(*e_pos)
        e_roll, e_pitch, e_yaw = [math.degrees(angle) for angle in e_rpy]
        e_tacview_yaw = (90.0 - e_yaw) % 360.0
        self.file.write(f"102,T={e_lon:.7f}|{e_lat:.7f}|{e_z:.1f}|{e_roll:.1f}|{e_pitch:.1f}|{e_tacview_yaw:.1f}\n")

    def close(self):
        """结束记录并关闭文件"""
        if self.file:
            self.file.close()
            self.is_recording = False
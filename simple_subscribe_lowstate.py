#!/usr/bin/env python3
"""
简化的 DDS rt/lowstate 订阅器
只订阅并打印接收到的原始消息信息
"""

import time
import signal
import sys

# 添加 unitree SDK 路径
sys.path.append('/home/unitree/Code/unitree_sdk2_python')

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC

class SimpleLowStateSubscriber:
    """简化的 LowState 订阅器"""

    def __init__(self):
        self.subscriber = None
        self.crc = CRC()
        self.message_count = 0

    def message_callback(self, msg: LowState_, datatype: str = ""):
        """处理接收到的 LowState 消息"""
        try:
            self.message_count += 1

            # 验证 CRC
            if self.crc.Crc(msg) != msg.crc:
                print(f"⚠️  CRC 校验失败！期望: {msg.crc}, 计算: {self.crc.Crc(msg)}")
                return

            # 直接打印原始消息信息
            print(f"\n📨 消息 #{self.message_count}:")
            print(f"  tick: {msg.tick}")
            print(f"  crc: {msg.crc}")

            # IMU 状态
            print(f"  imu_state:")
            print(f"    quaternion: {list(msg.imu_state.quaternion)}")
            print(f"    accelerometer: {list(msg.imu_state.accelerometer)}")
            print(f"    gyroscope: {list(msg.imu_state.gyroscope)}")

            # 电机状态
            motor_count = len(msg.motor_state)
            print(f"  motor_state: ({motor_count} 个电机)")

            for i, motor in enumerate(msg.motor_state):
                print(f"    motor[{i}]: q={motor.q}, dq={motor.dq}, tau_est={motor.tau_est}")

        except Exception as e:
            print(f"❌ 处理消息时出错: {e}")

    def start(self):
        """启动订阅器"""
        try:
            print("🔄 初始化 DDS 订阅器...")

            # 初始化 DDS 工厂 (0=实际模式, 1=仿真模式)
            ChannelFactoryInitialize(1)  # 使用仿真模式

            self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
            self.subscriber.Init(self.message_callback, 32)

            print("✅ DDS 订阅器已启动，正在监听 rt/lowstate 频道...")
            print("💡 按 Ctrl+C 停止监听")

            # 保持运行
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print(f"\n🛑 收到停止信号，已接收 {self.message_count} 条消息")
            self.stop()
        except Exception as e:
            print(f"❌ 启动订阅器失败: {e}")
            self.stop()

    def stop(self):
        """停止订阅器"""
        if self.subscriber:
            try:
                self.subscriber = None
                print("✅ DDS 订阅器已关闭")
            except Exception as e:
                print(f"⚠️  关闭订阅器时出错: {e}")

def main():
    """主函数"""
    print("🚀 简化的 DDS LowState 订阅器")
    print("=" * 50)

    # 检查是否安装了必要的包
    try:
        import unitree_sdk2py
    except ImportError:
        print("❌ 未安装 unitree_sdk2py，请先安装 Unitree SDK")
        print("   SDK 路径: /home/unitree/Code/unitree_sdk2_python")
        return

    # 创建订阅器并启动
    subscriber = SimpleLowStateSubscriber()

    def signal_handler(signum, frame):
        """信号处理函数"""
        print(f"\n🛑 收到信号 {signum}，正在退出...")
        subscriber.stop()
        sys.exit(0)

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 启动订阅器
    subscriber.start()

if __name__ == "__main__":
    main()

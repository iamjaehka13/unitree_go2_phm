#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import csv
import time
import os
from unitree_go.msg import LowState

CSV_FILE = "go2_full_log.csv"

# =========================
# Joint name mapping (12 DoF)
# =========================
JOINT_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",   # Leg 0
    "FL_hip", "FL_thigh", "FL_calf",   # Leg 1
    "RR_hip", "RR_thigh", "RR_calf",   # Leg 2
    "RL_hip", "RL_thigh", "RL_calf",   # Leg 3
]


class LowStateLogger(Node):
    def __init__(self):
        super().__init__('lowstate_full_logger')

        # 'w' 모드로 열어 실행할 때마다 새로 작성합니다. (이어서 쓰려면 'a'로 변경)
        self.csv_file = open(CSV_FILE, 'w', newline='')
        self.writer = csv.writer(self.csv_file)

        # 헤더 작성
        self.write_header()

        # LowState 구독
        self.subscription = self.create_subscription(
            LowState,
            '/lowstate',
            self.listener_callback,
            10
        )

        self.start_time = time.time()
        self.last_logged = 0.0

        self.get_logger().info(
            f"로깅 시작 | 저장 경로: {os.path.abspath(CSV_FILE)}"
        )

    # =========================
    # CSV Header
    # =========================
    def write_header(self):
        header = ["time_s"]

        # IMU
        header += ["imu_qw", "imu_qx", "imu_qy", "imu_qz"]
        header += ["imu_gyro_x", "imu_gyro_y", "imu_gyro_z"]
        header += ["imu_acc_x", "imu_acc_y", "imu_acc_z"]
        header += ["imu_rpy_x", "imu_rpy_y", "imu_rpy_z"]
        header += ["imu_temp"]

        # MotorState (12 joints only)
        for name in JOINT_NAMES:
            header += [
                f"{name}_mode",
                f"{name}_q",
                f"{name}_dq",
                f"{name}_ddq",
                f"{name}_tau_est",
                f"{name}_temp",
                f"{name}_lost",
            ]

        # BMS
        header += [
            "bms_version_high",
            "bms_version_low",
            "bms_status",
            "bms_soc",
            "bms_current",
            "bms_cycle",
            "bms_bq_ntc1",
            "bms_bq_ntc2",
            "bms_mcu_ntc1",
            "bms_mcu_ntc2",
        ]

        # BMS cell voltage (0~7 only)
        for i in range(8):
            header.append(f"bms_cell_vol_{i}")

        # Foot force
        header += ["foot_FL", "foot_FR", "foot_RL", "foot_RR"]
        header += ["foot_FL_est", "foot_FR_est", "foot_RL_est", "foot_RR_est"]

        # Tick
        header.append("tick")

        # Wireless remote
        for i in range(40):
            header.append(f"wireless_remote_{i}")

        # Other (fan_frequency 제거됨)
        header += [
            "bit_flag",
            "adc_reel",
            "temperature_ntc1",
            "temperature_ntc2",
            "power_v",
            "power_a",
        ]

        # Reserve & CRC
        header += ["reserve", "crc"]

        self.writer.writerow(header)
        self.csv_file.flush()

    # =========================
    # Callback
    # =========================
    def listener_callback(self, msg: LowState):
        now = time.time()

        # 1초 주기 로깅
        if now - self.last_logged < 1.0:
            return

        self.last_logged = now
        t = round(now - self.start_time, 2)

        try:
            row = [t]

            # IMU
            row += [round(v, 4) for v in msg.imu_state.quaternion]
            row += [round(v, 4) for v in msg.imu_state.gyroscope]
            row += [round(v, 4) for v in msg.imu_state.accelerometer]
            row += [round(v, 4) for v in msg.imu_state.rpy]
            row.append(msg.imu_state.temperature)

            # MotorState (0~11 only)
            for m in msg.motor_state[:12]:
                row += [
                    m.mode,
                    round(m.q, 4),
                    round(m.dq, 4),
                    round(m.ddq, 4),
                    round(m.tau_est, 4),
                    m.temperature,
                    m.lost,
                ]

            # BMS
            b = msg.bms_state
            row += [
                b.version_high,
                b.version_low,
                b.status,
                b.soc,
                b.current,
                b.cycle,
                b.bq_ntc[0],
                b.bq_ntc[1],
                b.mcu_ntc[0],
                b.mcu_ntc[1],
            ]
            row += [v for v in b.cell_vol[:8]]

            # Foot force
            row += [f for f in msg.foot_force]
            row += [f for f in msg.foot_force_est]

            # Tick
            row.append(msg.tick)

            # Wireless remote
            row += [int(w) for w in msg.wireless_remote]

            # Other
            row += [
                msg.bit_flag,
                msg.adc_reel,
                msg.temperature_ntc1,
                msg.temperature_ntc2,
                round(msg.power_v, 2),
                round(msg.power_a, 2),
            ]

            # Reserve & CRC
            row.append(msg.reserve)
            row.append(msg.crc)

            self.writer.writerow(row)
            self.csv_file.flush()

            self.get_logger().info(
                f"[저장] t={t}s | V={msg.power_v:.2f}V | I={msg.power_a:.2f}A"
            )

        except Exception as e:
            self.get_logger().error(f"CSV 기록 오류: {e}")

    # =========================
    # Shutdown
    # =========================
    def destroy_node(self):
        self.get_logger().info("CSV 파일 저장 후 노드 종료")
        self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LowStateLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


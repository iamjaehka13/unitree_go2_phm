#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from unitree_go.msg import LowState


# Joint name mapping (12 DoF)
JOINT_NAMES = [
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
]


class LowStateLogger(Node):
    def __init__(
        self,
        output_csv: str,
        topic: str,
        log_hz: float,
        flush_every_n: int,
        print_every_s: float,
    ):
        super().__init__("lowstate_full_logger")

        self.output_csv = output_csv
        self.log_period_s = (1.0 / float(log_hz)) if float(log_hz) > 0.0 else 0.0
        self.flush_every_n = max(int(flush_every_n), 1)
        self.print_every_s = max(float(print_every_s), 0.2)

        self.csv_file = open(self.output_csv, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self._write_header()

        # Use sensor-data QoS for high-rate robot telemetry.
        self.subscription = self.create_subscription(
            LowState,
            topic,
            self.listener_callback,
            qos_profile_sensor_data,
        )

        self.start_mono = time.monotonic()
        self.last_logged_mono = 0.0
        self.last_print_mono = self.start_mono
        self.rows_logged = 0
        self.callback_count = 0
        self.gated_count = 0

        self.get_logger().info(
            f"로깅 시작 | topic={topic} | target_hz={log_hz:.1f} | "
            f"저장 경로: {os.path.abspath(self.output_csv)}"
        )

    def _write_header(self):
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

        for i in range(8):
            header.append(f"bms_cell_vol_{i}")

        # Foot force
        header += ["foot_FL", "foot_FR", "foot_RL", "foot_RR"]
        header += ["foot_FL_est", "foot_FR_est", "foot_RL_est", "foot_RR_est"]

        header.append("tick")

        # Wireless remote
        for i in range(40):
            header.append(f"wireless_remote_{i}")

        # Other
        header += [
            "bit_flag",
            "adc_reel",
            "temperature_ntc1",
            "temperature_ntc2",
            "power_v",
            "power_a",
        ]

        header += ["reserve", "crc"]

        self.writer.writerow(header)
        self.csv_file.flush()

    def _maybe_print_stats(self, msg: LowState, now_mono: float):
        if (now_mono - self.last_print_mono) < self.print_every_s:
            return
        elapsed = max(now_mono - self.start_mono, 1e-6)
        log_hz = self.rows_logged / elapsed
        cb_hz = self.callback_count / elapsed
        self.get_logger().info(
            f"[상태] rows={self.rows_logged} ({log_hz:.1f}Hz) | "
            f"callbacks={self.callback_count} ({cb_hz:.1f}Hz) | "
            f"gated={self.gated_count} | V={msg.power_v:.2f} | I={msg.power_a:.2f}"
        )
        self.last_print_mono = now_mono

    def listener_callback(self, msg: LowState):
        now_mono = time.monotonic()
        self.callback_count += 1

        if self.log_period_s > 0.0 and (now_mono - self.last_logged_mono) < self.log_period_s:
            self.gated_count += 1
            return
        self.last_logged_mono = now_mono
        t = now_mono - self.start_mono

        try:
            row = [t]

            # IMU
            row += [float(v) for v in msg.imu_state.quaternion]
            row += [float(v) for v in msg.imu_state.gyroscope]
            row += [float(v) for v in msg.imu_state.accelerometer]
            row += [float(v) for v in msg.imu_state.rpy]
            row.append(int(msg.imu_state.temperature))

            # MotorState (0~11 only)
            for m in msg.motor_state[:12]:
                row += [
                    int(m.mode),
                    float(m.q),
                    float(m.dq),
                    float(m.ddq),
                    float(m.tau_est),
                    int(m.temperature),
                    int(m.lost),
                ]

            # BMS
            b = msg.bms_state
            row += [
                int(b.version_high),
                int(b.version_low),
                int(b.status),
                int(b.soc),
                int(b.current),
                int(b.cycle),
                int(b.bq_ntc[0]),
                int(b.bq_ntc[1]),
                int(b.mcu_ntc[0]),
                int(b.mcu_ntc[1]),
            ]
            row += [int(v) for v in b.cell_vol[:8]]

            # Foot force
            row += [int(f) for f in msg.foot_force]
            row += [int(f) for f in msg.foot_force_est]

            row.append(int(msg.tick))

            # Wireless remote
            row += [int(w) for w in msg.wireless_remote]

            # Other
            row += [
                int(msg.bit_flag),
                float(msg.adc_reel),
                int(msg.temperature_ntc1),
                int(msg.temperature_ntc2),
                float(msg.power_v),
                float(msg.power_a),
            ]

            row.append(int(msg.reserve))
            row.append(int(msg.crc))

            self.writer.writerow(row)
            self.rows_logged += 1

            if (self.rows_logged % self.flush_every_n) == 0:
                self.csv_file.flush()

            self._maybe_print_stats(msg, now_mono)

        except Exception as e:
            self.get_logger().error(f"CSV 기록 오류: {e}")

    def destroy_node(self):
        elapsed = max(time.monotonic() - self.start_mono, 1e-6)
        self.csv_file.flush()
        self.csv_file.close()
        self.get_logger().info(
            f"CSV 저장 후 종료 | rows={self.rows_logged} | "
            f"elapsed={elapsed:.2f}s | avg_log_hz={self.rows_logged/elapsed:.1f}"
        )
        super().destroy_node()


def _parse_args():
    parser = argparse.ArgumentParser(description="Unitree Go2 LowState CSV logger (ROS2)")
    parser.add_argument("--output_csv", type=str, default="go2_full_log.csv")
    parser.add_argument("--topic", type=str, default="/lowstate")
    parser.add_argument(
        "--log_hz",
        type=float,
        default=500.0,
        help="Target logging rate. Use 0 to log every callback.",
    )
    parser.add_argument("--flush_every_n", type=int, default=100)
    parser.add_argument("--print_every_s", type=float, default=1.0)
    return parser.parse_known_args()


def main():
    args, ros_args = _parse_args()
    rclpy.init(args=ros_args)
    node = LowStateLogger(
        output_csv=args.output_csv,
        topic=args.topic,
        log_hz=float(args.log_hz),
        flush_every_n=int(args.flush_every_n),
        print_every_s=float(args.print_every_s),
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

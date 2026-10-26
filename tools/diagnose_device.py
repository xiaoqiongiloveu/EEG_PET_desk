"""设备串口自检脚本。

用法：
    python tools/diagnose_device.py            # 自动找口并采集 5 秒
    python tools/diagnose_device.py COM3 -t 10
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import serial  # noqa: E402

from app.constants import (  # noqa: E402
    CMD_ENABLE_DEVICE,
    CMD_SET_CONFIG,
    CMD_START_STREAM,
    CMD_STOP_STREAM,
    TYPE_EEG,
)
from app.protocol import (  # noqa: E402
    FrameParser,
    build_command_frame,
    build_config_payload,
    parse_status_payload,
)
from app.transport.port_scan import list_serial_ports, resolve_port  # noqa: E402

TYPE_LABELS = {
    0x01: "EEG",
    0x02: "IMU",
    0x03: "阻抗",
    0x04: "状态",
    0x05: "辅助",
    0x11: "命令响应",
    0x20: "文件",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="NiceDog 设备串口自检")
    parser.add_argument("port", nargs="?", default="", help="串口名称，留空自动识别")
    parser.add_argument("-t", "--seconds", type=float, default=5.0, help="采集时长")
    parser.add_argument("-b", "--baudrate", type=int, default=460800, help="波特率")
    parser.add_argument("--rate", type=int, default=250, help="采样率")
    parser.add_argument("--gain", type=int, default=24, help="增益")
    args = parser.parse_args()

    ports = list_serial_ports()
    print("可用串口：")
    for port in ports:
        print(f"  {port.device:<8} {port.description}  {'[目标设备]' if port.is_target else ''}")
    port_name = resolve_port(args.port)
    if not port_name:
        print("未找到目标设备（VID:PID=0483:5740），请检查 USB 连接。")
        return 1

    print(f"打开 {port_name} @ {args.baudrate} ...")
    parser_ = FrameParser()
    sequence = 0
    counts: Counter = Counter()
    with serial.Serial(port_name, args.baudrate, timeout=0.05) as port:
        port.reset_input_buffer()

        def send(command: int, payload: bytes = b"") -> None:
            nonlocal sequence
            sequence = (sequence + 1) & 0xFFFF
            port.write(build_command_frame(command, sequence, payload))
            port.flush()

        send(CMD_ENABLE_DEVICE)
        time.sleep(0.1)
        send(CMD_SET_CONFIG, build_config_payload(args.rate, args.gain))
        time.sleep(0.1)
        send(CMD_START_STREAM)
        print(f"开始采集 {args.seconds:.0f} 秒 ...")

        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            data = port.read(port.in_waiting or 1)
            if not data:
                continue
            for frame in parser_.feed(data):
                counts[frame.packet_type] += 1
                if frame.packet_type == 0x04 and counts[frame.packet_type] == 1:
                    status = parse_status_payload(frame.payload)
                    print(
                        f"  状态：{status.state_code} 电量 {status.battery_v:.2f}V "
                        f"SD {status.sd_usage_percent}% 固件 {status.firmware}"
                    )
        send(CMD_STOP_STREAM)
        time.sleep(0.2)

    print("接收统计：")
    for packet_type, count in sorted(counts.items()):
        print(f"  {TYPE_LABELS.get(packet_type, f'0x{packet_type:02X}'):<8} {count}")
    print(f"CRC 错误 {parser_.crc_errors}，畸形帧 {parser_.malformed_frames}")
    if counts.get(TYPE_EEG, 0) == 0:
        print("未收到 EEG 数据，请检查设备是否已使能、是否被其他程序占用。")
        return 2
    expected = int(args.seconds * args.rate)
    print(f"EEG 帧 {counts[TYPE_EEG]}，理论约 {expected}（允许 ±5%）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

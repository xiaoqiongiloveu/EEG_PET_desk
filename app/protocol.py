"""ADS1299 USB-CDC 二进制协议编解码。

帧结构（与固件 binary_protocol.c 一致）：
    55 AA | LEN(2, LE) | TYPE(1) | SEQ(2, LE) | TS(4, LE) | PAYLOAD(N) | CRC16(2, LE)
其中 LEN = 7 + N；CRC 覆盖 TYPE、SEQ、TS 与 PAYLOAD。
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field

import numpy as np

from .constants import (
    SYNC_BYTES,
    TYPE_COMMAND,
    TYPE_EEG,
)

CRC_POLY = 0x1021
CRC_INIT = 0xFFFF


def crc16_ccitt(data: bytes) -> int:
    """CRC16-CCITT-FALSE（poly=0x1021, init=0xFFFF, refin/refout=False）。"""

    crc = CRC_INIT
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ CRC_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


@dataclass(frozen=True, slots=True)
class BinaryFrame:
    packet_type: int
    sequence: int
    device_timestamp_ms: int
    payload: bytes


class FrameParser:
    """增量式解析器，支持粘包、半包、垃圾字节与 CRC 错误重同步。"""

    def __init__(self, max_body_length: int = 4096) -> None:
        self.buffer = bytearray()
        self.max_body_length = max_body_length
        self.crc_errors = 0
        self.malformed_frames = 0

    def feed(self, data: bytes) -> list[BinaryFrame]:
        self.buffer.extend(data)
        frames: list[BinaryFrame] = []
        while True:
            frame = self._pop_one()
            if frame is None:
                break
            frames.append(frame)
        return frames

    def _pop_one(self) -> BinaryFrame | None:
        while True:
            sync_index = self.buffer.find(SYNC_BYTES)
            if sync_index < 0:
                # 保留最后一个字节，防止 0x55 恰好位于缓冲区末尾。
                self.buffer[:] = self.buffer[-1:]
                return None
            if sync_index:
                del self.buffer[:sync_index]
            if len(self.buffer) < 4:
                return None
            body_length = int.from_bytes(self.buffer[2:4], "little")
            if body_length < 7 or body_length > self.max_body_length:
                self.malformed_frames += 1
                del self.buffer[:2]
                continue
            total_length = 4 + body_length + 2
            if len(self.buffer) < total_length:
                return None
            raw = bytes(self.buffer[:total_length])
            del self.buffer[:total_length]
            expected = int.from_bytes(raw[-2:], "little")
            if crc16_ccitt(raw[4:-2]) != expected:
                self.crc_errors += 1
                continue
            return BinaryFrame(
                packet_type=raw[4],
                sequence=int.from_bytes(raw[5:7], "little"),
                device_timestamp_ms=int.from_bytes(raw[7:11], "little"),
                payload=raw[11:-2],
            )


def build_command_frame(command_id: int, sequence: int, payload: bytes = b"") -> bytes:
    """构造一条上位机命令帧。"""

    command_payload = bytes([command_id & 0xFF]) + payload
    timestamp = int(time.monotonic() * 1000) & 0xFFFFFFFF
    sequence &= 0xFFFF
    body_length = 7 + len(command_payload)
    body = struct.pack("<BHI", TYPE_COMMAND, sequence, timestamp) + command_payload
    header = struct.pack("<HHBHI", 0xAA55, body_length, TYPE_COMMAND, sequence, timestamp)
    return header + command_payload + struct.pack("<H", crc16_ccitt(body))


def build_config_payload(
    sample_rate: int,
    gain: int,
    channel_mask: int = 0xFF,
    input_type: int = 0,
    bias_enabled: int = 1,
    test_signal: int = 0,
    duration_s: int = 0,
) -> bytes:
    """SET_CONF 的 11 字节参数（不含 cmd_id）。"""

    return struct.pack(
        "<HBBBBBI",
        int(sample_rate) & 0xFFFF,
        int(gain) & 0xFF,
        int(channel_mask) & 0xFF,
        int(input_type) & 0xFF,
        int(bias_enabled) & 0xFF,
        int(test_signal) & 0xFF,
        int(duration_s) & 0xFFFFFFFF,
    )


@dataclass(slots=True)
class EegPayload:
    values_uv: np.ndarray
    lead_off: np.ndarray
    acc_g: np.ndarray
    gyro_dps: np.ndarray


def parse_eeg_payload(payload: bytes, gain: int) -> EegPayload:
    """解析 47 字节 EEG payload。"""

    if len(payload) < 35:
        raise ValueError(f"EEG payload 过短: {len(payload)}")
    byte0, byte1 = payload[0], payload[1]
    lead_off = np.array(
        [
            bool(byte1 & 0x10),
            bool(byte1 & 0x20),
            bool(byte1 & 0x40),
            bool(byte1 & 0x80),
            bool(byte0 & 0x01),
            bool(byte0 & 0x02),
            bool(byte0 & 0x04),
            bool(byte0 & 0x08),
        ],
        dtype=bool,
    )
    raw = np.frombuffer(payload, dtype="<i4", count=8, offset=3).astype(np.float64)
    values = raw * (4_500_000.0 / (max(int(gain), 1) * 8_388_607.0))
    if len(payload) >= 47:
        imu = np.frombuffer(payload, dtype="<i2", count=6, offset=35).astype(np.float64)
        acc = imu[:3] / 1000.0
        gyro = imu[3:] / 10.0
    else:
        acc = np.zeros(3, dtype=np.float64)
        gyro = np.zeros(3, dtype=np.float64)
    return EegPayload(values.astype(np.float32), lead_off, acc, gyro)


def parse_impedance_payload(payload: bytes) -> np.ndarray:
    if len(payload) < 16:
        raise ValueError(f"阻抗 payload 过短: {len(payload)}")
    return np.frombuffer(payload, dtype="<u2", count=8).astype(np.float32)


@dataclass(slots=True)
class AuxPayload:
    acc_g: np.ndarray
    gyro_dps: np.ndarray
    battery_v: float
    motion_score: int
    reserved: int = 0


def parse_aux_payload(payload: bytes) -> AuxPayload:
    if len(payload) < 16:
        raise ValueError(f"AUX payload 过短: {len(payload)}")
    imu = np.frombuffer(payload, dtype="<i2", count=6, offset=0).astype(np.float64)
    battery = int.from_bytes(payload[12:14], "little") / 100.0
    return AuxPayload(
        acc_g=(imu[:3] / 1000.0).astype(np.float32),
        gyro_dps=(imu[3:] / 10.0).astype(np.float32),
        battery_v=battery,
        motion_score=int(payload[14]),
        reserved=int(payload[15]),
    )


@dataclass(slots=True)
class DeviceStatus:
    device_type: int = 1
    sd_usage_percent: int = 0
    sd_ok: bool = True
    battery_v: float = 0.0
    state_code: int = 0
    task_id: int = 0
    pb_status: int = 0
    firmware: str = ""
    serial_no: str = ""
    raw: bytes = field(default=b"", repr=False)


def parse_status_payload(payload: bytes) -> DeviceStatus:
    """兼容固件当前布局与旧版文档布局。"""

    if len(payload) >= 11:
        battery = int.from_bytes(payload[3:5], "little") / 100.0
        blob = payload[11:]
        if b"\x00" in blob:
            ver_raw, sn_raw = blob.split(b"\x00", 1)
        else:
            ver_raw, sn_raw = blob, b""
        return DeviceStatus(
            device_type=payload[0],
            sd_usage_percent=payload[1],
            sd_ok=bool(payload[2]),
            battery_v=battery,
            state_code=payload[5],
            task_id=int.from_bytes(payload[6:10], "little"),
            pb_status=payload[10],
            firmware=ver_raw.decode("ascii", errors="replace"),
            serial_no=sn_raw.decode("ascii", errors="replace"),
            raw=payload,
        )
    if len(payload) >= 5:
        return DeviceStatus(
            sd_ok=bool(payload[2]),
            battery_v=int.from_bytes(payload[3:5], "little") / 1000.0,
            raw=payload,
        )
    return DeviceStatus(raw=payload)


@dataclass(frozen=True, slots=True)
class CommandResponse:
    command_id: int
    status: int
    message: str

    @property
    def ok(self) -> bool:
        return self.status == 0


def parse_command_response(payload: bytes) -> CommandResponse:
    if len(payload) < 2:
        raise ValueError("命令响应 payload 过短")
    message = payload[2:].decode("utf-8", errors="replace").strip("\x00")
    return CommandResponse(payload[0], payload[1], message)


def pack_frame(
    packet_type: int,
    sequence: int,
    device_timestamp_ms: int,
    payload: bytes,
) -> bytes:
    """测试与模拟器使用的通用打包函数。"""

    sequence &= 0xFFFF
    timestamp = device_timestamp_ms & 0xFFFFFFFF
    body_length = 7 + len(payload)
    body = struct.pack("<BHI", packet_type, sequence, timestamp) + payload
    header = struct.pack("<HHBHI", 0xAA55, body_length, packet_type, sequence, timestamp)
    return header + payload + struct.pack("<H", crc16_ccitt(body))


def eeg_packet(sample: np.ndarray, sequence: int, timestamp_ms: int, gain: int) -> bytes:
    """把 8 个 µV 值打包成 EEG 帧（仅用于模拟器与测试）。"""

    raw = np.rint(np.asarray(sample, dtype=np.float64) * (gain * 8_388_607.0) / 4_500_000.0)
    raw = np.clip(raw, -8_388_608, 8_388_607).astype("<i4")
    payload = bytearray(47)
    payload[3:35] = raw.tobytes()
    return pack_frame(TYPE_EEG, sequence, timestamp_ms, bytes(payload))

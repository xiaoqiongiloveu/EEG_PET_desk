"""采集层与界面层之间传递的数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class SampleBatch:
    """一批 EEG 采样点，通道顺序与 constants.CHANNELS_RAW 一致。"""

    values: np.ndarray  # (n, 8) float32，单位 µV
    timestamps: np.ndarray  # (n,) float64，主机单调时钟秒
    device_ms: np.ndarray  # (n,) uint32，设备时间戳
    lead_off: np.ndarray  # (n, 8) bool
    sequences: np.ndarray  # (n,) uint16

    @property
    def count(self) -> int:
        return int(self.values.shape[0])


@dataclass(slots=True)
class AcquisitionStats:
    """每秒推送一次的采集统计（累计值，从本次连接建立开始）。"""

    frames: int = 0
    samples: int = 0
    bytes_received: int = 0
    crc_errors: int = 0
    malformed_frames: int = 0
    # 序号缺口口径：按 EEG 帧序号跳变推算的丢样数（设备已发出但没有到达主机）。
    sequence_gaps: int = 0
    lost_by_seq: int = 0
    # 设备时间戳口径：按帧时间戳间隔推算的丢样数（含采样/缓冲端丢包）。
    lost_by_time: int = 0
    # lost_by_seq / (已收 + lost_by_seq)
    loss_rate: float = 0.0
    # 断流次数（序号或时间戳跳变超出合理范围，通常为设备重启/链路中断）。
    outages: int = 0
    # 设备端上报的计数器（需固件支持，见 STATUS 扩展块）。
    device_drops: int = 0
    device_link_errors: int = 0
    device_report_valid: bool = False
    frame_rate: float = 0.0
    sample_rate: float = 0.0
    elapsed_s: float = 0.0
    packet_counts: dict[int, int] = field(default_factory=dict)

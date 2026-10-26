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
    """每秒推送一次的采集统计。"""

    frames: int = 0
    samples: int = 0
    bytes_received: int = 0
    crc_errors: int = 0
    malformed_frames: int = 0
    sequence_gaps: int = 0
    dropped_samples: int = 0
    frame_rate: float = 0.0
    sample_rate: float = 0.0
    elapsed_s: float = 0.0
    packet_counts: dict[int, int] = field(default_factory=dict)

"""时域统计指标。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats


@dataclass(slots=True)
class ChannelStats:
    mean_uv: float
    std_uv: float
    rms_uv: float
    p2p_uv: float
    skewness: float
    kurtosis: float


def channel_statistics(values: np.ndarray) -> list[ChannelStats]:
    """按列计算均值、标准差、RMS、峰峰值、偏度与峰度。"""

    data = np.asarray(values, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    results: list[ChannelStats] = []
    for idx in range(data.shape[1]):
        column = data[:, idx]
        results.append(
            ChannelStats(
                mean_uv=float(np.mean(column)),
                std_uv=float(np.std(column)),
                rms_uv=float(np.sqrt(np.mean(column**2))),
                p2p_uv=float(np.ptp(column)),
                skewness=float(scipy_stats.skew(column)) if column.size > 2 else 0.0,
                kurtosis=float(scipy_stats.kurtosis(column)) if column.size > 3 else 0.0,
            )
        )
    return results


def rms(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.nan_to_num(data) ** 2)))

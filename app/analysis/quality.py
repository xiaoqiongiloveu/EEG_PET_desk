"""信号质量评估与伪迹标记。

质量门控只使用可解释的时域/频域指标，不依赖训练模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .spectrum import welch_psd

# 判定阈值（单位 µV，除注明外）。
SATURATION_UV = 150_000.0
FLAT_STD_UV = 0.5
DRIFT_RATIO = 3.0
HIGH_FREQ_RATIO = 0.45
MAX_CORRELATION = 0.995


@dataclass(slots=True)
class ChannelQuality:
    index: int
    name: str
    rms_uv: float
    p2p_uv: float
    drift_uv: float
    high_freq_ratio: float
    correlation: float
    score: int
    flags: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.score >= 60


def _channel_correlation(data: np.ndarray) -> np.ndarray:
    """每个通道与其余通道均值的相关系数，用于发现短接或串扰。"""

    channels = data.shape[1]
    result = np.zeros(channels, dtype=np.float64)
    if channels < 2:
        return result
    for idx in range(channels):
        others = np.delete(data, idx, axis=1).mean(axis=1)
        a = data[:, idx]
        if np.std(a) < 1e-9 or np.std(others) < 1e-9:
            continue
        result[idx] = float(np.corrcoef(a, others)[0, 1])
    return result


def evaluate_channels(
    values: np.ndarray,
    fs: float,
    channel_names: list[str] | tuple[str, ...],
    saturation_uv: float = SATURATION_UV,
) -> list[ChannelQuality]:
    """逐通道计算质量评分与问题标签。"""

    data = np.asarray(values, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    n_samples, channels = data.shape

    freqs, psd = welch_psd(data, fs) if n_samples >= 16 else (np.zeros(1), np.zeros((channels, 1)))
    band_mask = (freqs >= 0.5) & (freqs <= min(45.0, fs / 2.0))
    high_mask = (freqs >= 40.0) & (freqs <= min(100.0, fs / 2.0))
    correlations = _channel_correlation(data)

    results: list[ChannelQuality] = []
    for idx in range(channels):
        column = data[:, idx]
        std = float(np.std(column))
        p2p = float(np.ptp(column))
        rms = float(np.sqrt(np.mean(column**2)))
        if n_samples >= 10:
            edge = max(n_samples // 10, 1)
            drift = abs(float(np.mean(column[-edge:]) - np.mean(column[:edge])))
        else:
            drift = 0.0

        total = float(np.sum(psd[idx, band_mask])) if band_mask.any() else 0.0
        high = float(np.sum(psd[idx, high_mask])) if high_mask.any() else 0.0
        high_ratio = high / total if total > 0 else 0.0

        flags: list[str] = []
        score = 100
        if std < FLAT_STD_UV:
            flags.append("平线")
            score -= 60
        if np.any(np.abs(column) >= saturation_uv):
            flags.append("饱和")
            score -= 40
        if std > 0 and drift / std > DRIFT_RATIO:
            flags.append("漂移")
            score -= 20
        if high_ratio > HIGH_FREQ_RATIO:
            flags.append("高频噪声")
            score -= 25
        if abs(correlations[idx]) > MAX_CORRELATION:
            flags.append("通道相关性过高")
            score -= 20
        if p2p > 4 * saturation_uv:
            flags.append("幅度异常")
            score -= 10

        name = channel_names[idx] if idx < len(channel_names) else f"CH{idx + 1}"
        results.append(
            ChannelQuality(
                index=idx,
                name=str(name),
                rms_uv=rms,
                p2p_uv=p2p,
                drift_uv=drift,
                high_freq_ratio=high_ratio,
                correlation=float(correlations[idx]),
                score=max(score, 0),
                flags=flags,
            )
        )
    return results


def artifact_mask(
    values: np.ndarray,
    sigma: float = 6.0,
    min_uv: float = 80.0,
) -> np.ndarray:
    """基于稳健 z 分数标记大幅瞬态伪迹（眨眼、运动等），返回逐采样布尔掩码。"""

    data = np.asarray(values, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    median = np.median(data, axis=0, keepdims=True)
    mad = np.median(np.abs(data - median), axis=0, keepdims=True) * 1.4826
    mad = np.where(mad < 1e-6, 1e-6, mad)
    z = np.abs(data - median) / mad
    return np.any((z > sigma) & (np.abs(data - median) > min_uv), axis=1)


def artifact_ratio(values: np.ndarray, sigma: float = 6.0, min_uv: float = 80.0) -> float:
    mask = artifact_mask(values, sigma=sigma, min_uv=min_uv)
    return float(np.mean(mask)) if mask.size else 0.0

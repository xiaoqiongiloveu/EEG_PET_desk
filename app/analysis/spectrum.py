"""频谱分析：Welch PSD、频带功率、SEF95、频谱重心与 alpha 峰。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

# 默认分析频带（Hz）。
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass(slots=True)
class SpectrumResult:
    freqs: np.ndarray
    psd: np.ndarray  # (channels, freqs)，单位 µV²/Hz
    band_power: dict[str, np.ndarray] = field(default_factory=dict)
    relative_power: dict[str, np.ndarray] = field(default_factory=dict)
    total_power: np.ndarray | None = None
    sef95: np.ndarray | None = None
    centroid: np.ndarray | None = None
    alpha_peak_hz: np.ndarray | None = None
    alpha_peak_power: np.ndarray | None = None


def _as_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    return array


def welch_psd(
    values: np.ndarray,
    fs: float,
    nperseg_s: float = 2.0,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """计算多通道 Welch 功率谱，返回 (freqs, psd[channels, freqs])。"""

    data = _as_2d(values)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    n_samples = data.shape[0]
    if n_samples < 8:
        raise ValueError("样本过少，无法计算功率谱")
    nperseg = int(min(max(fs * nperseg_s, 16), n_samples))
    noverlap = int(nperseg * overlap) if nperseg > 2 else 0
    freqs, psd = signal.welch(
        data,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        axis=0,
        scaling="density",
    )
    return freqs, np.transpose(psd, (1, 0))


def band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: dict[str, tuple[float, float]] | None = None,
) -> dict[str, np.ndarray]:
    """对各频带做梯形积分，返回 {band: (channels,)}。"""

    bands = bands or BANDS
    result: dict[str, np.ndarray] = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            result[name] = np.zeros(psd.shape[0], dtype=np.float64)
            continue
        result[name] = np.trapezoid(psd[:, mask], freqs[mask], axis=1)
    return result


def total_power(freqs: np.ndarray, psd: np.ndarray, low: float = 0.5, high: float | None = None) -> np.ndarray:
    mask = freqs >= low
    if high is not None:
        mask &= freqs <= high
    if not np.any(mask):
        return np.zeros(psd.shape[0], dtype=np.float64)
    return np.trapezoid(psd[:, mask], freqs[mask], axis=1)


def relative_power(
    band: dict[str, np.ndarray], total: np.ndarray
) -> dict[str, np.ndarray]:
    safe_total = np.where(total > 0, total, np.nan)
    return {name: (power / safe_total) for name, power in band.items()}


def spectral_edge_frequency(
    freqs: np.ndarray, psd: np.ndarray, ratio: float = 0.95, low: float = 0.5
) -> np.ndarray:
    """谱边缘频率 SEF（默认 SEF95），单位 Hz。"""

    mask = freqs >= low
    sub_freqs = freqs[mask]
    sub_psd = psd[:, mask]
    cumulative = np.cumsum(sub_psd, axis=1)
    totals = cumulative[:, -1:]
    target = totals * ratio
    indices = np.argmax(cumulative >= target, axis=1)
    return sub_freqs[np.clip(indices, 0, len(sub_freqs) - 1)]


def spectral_centroid(freqs: np.ndarray, psd: np.ndarray, low: float = 0.5) -> np.ndarray:
    mask = freqs >= low
    sub_freqs = freqs[mask]
    sub_psd = psd[:, mask]
    totals = sub_psd.sum(axis=1)
    safe = np.where(totals > 0, totals, np.nan)
    return (sub_psd * sub_freqs).sum(axis=1) / safe


def alpha_peak(
    freqs: np.ndarray,
    psd: np.ndarray,
    low: float = 7.5,
    high: float = 12.5,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 alpha 频段峰值频率与峰值功率。"""

    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        zeros = np.zeros(psd.shape[0], dtype=np.float64)
        return zeros, zeros
    sub_freqs = freqs[mask]
    sub_psd = psd[:, mask]
    indices = np.argmax(sub_psd, axis=1)
    return sub_freqs[indices], sub_psd[np.arange(sub_psd.shape[0]), indices]


def analyze_spectrum(
    values: np.ndarray,
    fs: float,
    nperseg_s: float = 2.0,
    bands: dict[str, tuple[float, float]] | None = None,
) -> SpectrumResult:
    """一次性完成常用的频谱指标计算。"""

    freqs, psd = welch_psd(values, fs, nperseg_s=nperseg_s)
    bands = bands or BANDS
    powers = band_power(freqs, psd, bands)
    total = total_power(freqs, psd, high=max(high for _, high in bands.values()))
    peak_hz, peak_power = alpha_peak(freqs, psd)
    return SpectrumResult(
        freqs=freqs,
        psd=psd,
        band_power=powers,
        relative_power=relative_power(powers, total),
        total_power=total,
        sef95=spectral_edge_frequency(freqs, psd, 0.95),
        centroid=spectral_centroid(freqs, psd),
        alpha_peak_hz=peak_hz,
        alpha_peak_power=peak_power,
    )

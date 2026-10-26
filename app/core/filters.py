"""实时与离线滤波、重参考。"""

from __future__ import annotations

import numpy as np
from scipy import signal

# 显示滤波使用 4 阶 Butterworth，足够平滑且相位延迟可控。
FILTER_ORDER = 4


def _nyquist(fs: float) -> float:
    return max(float(fs) / 2.0, 1.0)


def _safe_band(fs: float, low: float, high: float) -> tuple[float, float]:
    nyq = _nyquist(fs)
    low = max(0.01, float(low))
    high = min(float(high), nyq * 0.95)
    if high <= low * 1.05:
        low = max(0.01, high / 2.0)
    return low, high


def design_sos(fs: float, low: float, high: float) -> np.ndarray:
    """设计带通/高通/低通 SOS，自动适配采样率。"""

    nyq = _nyquist(fs)
    low, high = _safe_band(fs, low, high)
    if low <= 0.02:
        return signal.butter(FILTER_ORDER, high, btype="lowpass", fs=fs, output="sos")
    if high >= nyq * 0.94:
        return signal.butter(FILTER_ORDER, low, btype="highpass", fs=fs, output="sos")
    return signal.butter(FILTER_ORDER, [low, high], btype="bandpass", fs=fs, output="sos")


def design_notch(fs: float, freq: float, q: float = 30.0) -> np.ndarray:
    if not freq or freq <= 0 or freq >= _nyquist(fs) * 0.95:
        return np.empty((0, 6), dtype=np.float64)
    b, a = signal.iirnotch(float(freq), q, fs=fs)
    return signal.tf2sos(b, a)


class RealtimeFilter:
    """带状态的因果滤波器，用于实时显示。"""

    def __init__(self, fs: float, channels: int) -> None:
        self.fs = float(fs)
        self.channels = int(channels)
        self.low = 0.5
        self.high = 45.0
        self.notch = 50
        self.enabled = True
        self._sos: np.ndarray | None = None
        self._zi: np.ndarray | None = None
        self._notch_sos: np.ndarray | None = None
        self._notch_zi: np.ndarray | None = None
        self._initialized = False

    def configure(
        self,
        fs: float | None = None,
        low: float = 0.5,
        high: float = 45.0,
        notch: int = 50,
        enabled: bool = True,
    ) -> None:
        new_fs = float(fs) if fs is not None else self.fs
        changed = (
            abs(new_fs - self.fs) > 1e-6
            or float(low) != self.low
            or float(high) != self.high
            or int(notch) != self.notch
        )
        self.fs = new_fs
        self.low, self.high, self.notch, self.enabled = float(low), float(high), int(notch), bool(enabled)
        if not changed and self._sos is not None:
            return
        self._sos = design_sos(self.fs, self.low, self.high)
        self._notch_sos = design_notch(self.fs, self.notch)
        self._zi = None
        self._notch_zi = None
        self._initialized = False

    def reset(self) -> None:
        self._zi = None
        self._notch_zi = None
        self._initialized = False

    def process(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if not self.enabled or len(values) == 0:
            return values.astype(np.float32)
        if self._sos is None:
            self.configure(self.fs, self.low, self.high, self.notch, self.enabled)

        if self._zi is None:
            self._zi = signal.sosfilt_zi(self._sos)[:, :, None] * values[0][None, None, :]
        if self._notch_sos is not None and len(self._notch_sos) and self._notch_zi is None:
            self._notch_zi = signal.sosfilt_zi(self._notch_sos)[:, :, None] * values[0][None, None, :]

        filtered, self._zi = signal.sosfilt(self._sos, values, axis=0, zi=self._zi)
        if self._notch_sos is not None and len(self._notch_sos) and self._notch_zi is not None:
            filtered, self._notch_zi = signal.sosfilt(
                self._notch_sos, filtered, axis=0, zi=self._notch_zi
            )
        return filtered.astype(np.float32)


def offline_filter(
    values: np.ndarray,
    fs: float,
    low: float = 0.5,
    high: float = 45.0,
    notch: int = 50,
) -> np.ndarray:
    """零相位滤波，供回放分析使用。"""

    data = np.asarray(values, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if len(data) < 8:
        return data.astype(np.float32)
    sos = design_sos(fs, low, high)
    try:
        out = signal.sosfiltfilt(sos, data, axis=0)
    except ValueError:
        out = signal.sosfilt(sos, data, axis=0)
    notch_sos = design_notch(fs, notch)
    if len(notch_sos):
        try:
            out = signal.sosfiltfilt(notch_sos, out, axis=0)
        except ValueError:
            out = signal.sosfilt(notch_sos, out, axis=0)
    return out.astype(np.float32)


def remove_dc(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    if len(data) == 0:
        return data
    return data - np.median(data, axis=0, keepdims=True)


def apply_rereference(
    values: np.ndarray,
    channel_names: list[str],
    mode: str = "none",
    reference_channel: str = "",
    factor: float = 0.5,
) -> np.ndarray:
    """重参考。

    mode:
        none    保持设备原始参考
        car     共同平均参考
        channel 减去指定参考通道的 factor 倍（默认 0.5，对应双耳垂平均习惯）
    """

    data = np.asarray(values, dtype=np.float32)
    if data.ndim != 2 or len(data) == 0:
        return data
    normalized = mode.strip().lower()
    if normalized == "car":
        return (data - data.mean(axis=1, keepdims=True)).astype(np.float32)
    if normalized == "channel" and reference_channel:
        lookup = {name.upper(): i for i, name in enumerate(channel_names)}
        index = lookup.get(reference_channel.upper())
        if index is not None:
            return (data - data[:, [index]] * float(factor)).astype(np.float32)
    return data.copy()

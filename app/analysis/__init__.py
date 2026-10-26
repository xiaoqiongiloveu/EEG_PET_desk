"""信号分析：频谱、统计与质量评估。"""

from .quality import ChannelQuality, artifact_mask, artifact_ratio, evaluate_channels
from .spectrum import (
    BANDS,
    SpectrumResult,
    analyze_spectrum,
    band_power,
    spectral_edge_frequency,
)
from .stats import ChannelStats, channel_statistics, rms

__all__ = [
    "BANDS",
    "SpectrumResult",
    "analyze_spectrum",
    "band_power",
    "spectral_edge_frequency",
    "ChannelQuality",
    "evaluate_channels",
    "artifact_mask",
    "artifact_ratio",
    "ChannelStats",
    "channel_statistics",
    "rms",
]

"""用户设置持久化。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .constants import (
    CHANNELS_RAW,
    DEFAULT_BAUDRATE,
    DEFAULT_FILTER_HIGH,
    DEFAULT_FILTER_LOW,
    DEFAULT_NOTCH,
    DEFAULT_SCALE_UV,
    DEFAULT_WIFI_BAUDRATE,
    DEFAULT_WINDOW_S,
    DISPLAY_ORDER,
    EEG_GAIN,
    SAMPLE_RATE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
SETTINGS_PATH = CONFIG_DIR / "settings.json"


@dataclass
class Settings:
    port: str = ""
    baudrate: int = DEFAULT_BAUDRATE
    wifi_mode: bool = False
    wifi_baudrate: int = DEFAULT_WIFI_BAUDRATE
    source: str = "serial"
    auto_connect: bool = False
    auto_start: bool = False

    sample_rate: int = SAMPLE_RATE
    gain: int = EEG_GAIN
    channel_names: list[str] = field(default_factory=lambda: list(CHANNELS_RAW))
    display_order: list[int] = field(default_factory=lambda: list(DISPLAY_ORDER))

    storage_dir: str = ""
    window_s: float = DEFAULT_WINDOW_S
    scale_uv: float = DEFAULT_SCALE_UV
    filter_enabled: bool = True
    filter_low: float = DEFAULT_FILTER_LOW
    filter_high: float = DEFAULT_FILTER_HIGH
    notch: int = DEFAULT_NOTCH

    replay_window_s: float = DEFAULT_WINDOW_S
    replay_filter_enabled: bool = True
    replay_filter_low: float = DEFAULT_FILTER_LOW
    replay_filter_high: float = DEFAULT_FILTER_HIGH
    replay_notch: int = DEFAULT_NOTCH
    reref_mode: str = "none"
    reref_channel: str = ""

    def resolved_storage_dir(self) -> Path:
        path = Path(self.storage_dir) if self.storage_dir else PROJECT_ROOT / "sessions"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def normalized_channels(self) -> list[str]:
        names = [str(name).strip() or f"CH{i + 1}" for i, name in enumerate(self.channel_names)]
        if len(names) != 8:
            names = list(CHANNELS_RAW)
        return names

    def normalized_order(self) -> list[int]:
        order = [int(i) for i in self.display_order if 0 <= int(i) < 8]
        if len(order) != 8 or len(set(order)) != 8:
            order = list(DISPLAY_ORDER)
        return order


def load_settings(path: Path = SETTINGS_PATH) -> Settings:
    settings = Settings()
    if not path.exists():
        return settings
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return settings
    for key, value in data.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    return settings


def save_settings(settings: Settings, path: Path = SETTINGS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(settings), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

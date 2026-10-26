"""历史会话读取。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..constants import CHANNELS_RAW

SAMPLE_DTYPE = np.dtype("<f4")
TIMESTAMP_DTYPE = np.dtype("<f8")


@dataclass
class SessionInfo:
    path: Path
    metadata: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def sample_rate(self) -> int:
        return int(self.metadata.get("sample_rate", 250))

    @property
    def gain(self) -> int:
        return int(self.metadata.get("gain", 24))

    @property
    def channel_names(self) -> list[str]:
        names = self.metadata.get("channel_names") or list(CHANNELS_RAW)
        return [str(name) for name in names]

    @property
    def sample_count(self) -> int:
        return int(self.metadata.get("sample_count", 0))

    @property
    def created_at(self) -> str:
        return str(self.metadata.get("created_at", ""))

    @property
    def duration_s(self) -> float:
        if self.sample_count and self.sample_rate:
            return self.sample_count / self.sample_rate
        return float(self.metadata.get("duration_s", 0.0))

    def summary(self) -> str:
        created = self.created_at.replace("T", " ")[:19]
        return f"{created}  {self.duration_s:7.1f}s  {self.sample_rate}Hz  {len(self.channel_names)}ch"


def load_session(path: Path) -> SessionInfo:
    meta_path = path / "session.json"
    metadata: dict = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    if "sample_count" not in metadata:
        samples_path = path / "samples.bin"
        if samples_path.exists():
            channels = len(metadata.get("channel_names") or CHANNELS_RAW)
            metadata["sample_count"] = samples_path.stat().st_size // (4 * max(channels, 1))
    return SessionInfo(path=path, metadata=metadata)


def list_sessions(root: Path) -> list[SessionInfo]:
    if not root.exists():
        return []
    sessions = []
    for entry in sorted(root.iterdir(), reverse=True):
        if entry.is_dir() and (entry / "samples.bin").exists():
            sessions.append(load_session(entry))
    return sessions


class SessionReader:
    """按块读取大文件，避免一次性载入内存。"""

    def __init__(self, path: Path) -> None:
        self.info = load_session(path)
        self.path = path
        self.channels = len(self.info.channel_names)
        self.sample_rate = max(self.info.sample_rate, 1)
        self._samples = path / "samples.bin"
        self._timestamps = path / "timestamps.bin"
        frame_bytes = 4 * self.channels
        self.sample_count = (
            self._samples.stat().st_size // frame_bytes if self._samples.exists() else 0
        )

    @property
    def duration_s(self) -> float:
        return self.sample_count / self.sample_rate

    def read(self, start: int = 0, count: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """读取指定区间的采样与时间戳，时间戳单位为秒。"""

        start = max(int(start), 0)
        if count is None:
            count = self.sample_count - start
        count = max(min(int(count), self.sample_count - start), 0)
        if count == 0:
            return (
                np.empty((0, self.channels), dtype=np.float32),
                np.empty((0,), dtype=np.float64),
            )
        values = np.fromfile(
            self._samples,
            dtype=SAMPLE_DTYPE,
            count=count * self.channels,
            offset=start * 4 * self.channels,
        )
        values = values.reshape(-1, self.channels).astype(np.float32, copy=False)
        timestamps = self._read_timestamps(start, count)
        return values, timestamps

    def read_time_range(self, start_s: float, end_s: float) -> tuple[np.ndarray, np.ndarray]:
        start = int(max(start_s, 0.0) * self.sample_rate)
        count = int(max(end_s - start_s, 0.0) * self.sample_rate)
        return self.read(start, count)

    def _read_timestamps(self, start: int, count: int) -> np.ndarray:
        if self._timestamps.exists():
            data = np.fromfile(
                self._timestamps,
                dtype=TIMESTAMP_DTYPE,
                count=count,
                offset=start * 8,
            )
            if len(data) == count:
                return data.astype(np.float64, copy=False)
        return np.arange(start, start + count, dtype=np.float64) / self.sample_rate

    def read_events(self) -> list[dict]:
        path = self.path / "events.jsonl"
        if not path.exists():
            return []
        events: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def read_commands(self) -> list[str]:
        path = self.path / "commands.log"
        if not path.exists():
            return []
        return path.read_text(encoding="utf-8").splitlines()

    def read_impedance(self) -> np.ndarray:
        """返回 (n, 1 + channels)：第一列为相对时间秒。"""

        path = self.path / "impedance.tsv"
        if not path.exists():
            return np.empty((0, 1 + self.channels), dtype=np.float64)
        rows: list[list[float]] = []
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split("\t")
            try:
                rows.append([float(value) for value in parts[: 1 + self.channels]])
            except ValueError:
                continue
        if not rows:
            return np.empty((0, 1 + self.channels), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)

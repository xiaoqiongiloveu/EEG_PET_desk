"""固定容量多通道环形缓冲区。"""

from __future__ import annotations

import numpy as np


class MultiChannelRingBuffer:
    def __init__(self, capacity: int, channels: int) -> None:
        self.capacity = int(capacity)
        self.channels = int(channels)
        self.values = np.zeros((self.capacity, self.channels), dtype=np.float32)
        self.timestamps = np.zeros(self.capacity, dtype=np.float64)
        self._write = 0
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def clear(self) -> None:
        self._write = 0
        self._count = 0

    def append(self, values: np.ndarray, timestamps: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.channels:
            raise ValueError("环形缓冲区通道数不匹配")
        n = min(len(values), self.capacity)
        if n <= 0:
            return
        values = values[-n:]
        timestamps = timestamps[-n:]
        first = min(n, self.capacity - self._write)
        self.values[self._write : self._write + first] = values[:first]
        self.timestamps[self._write : self._write + first] = timestamps[:first]
        rest = n - first
        if rest:
            self.values[:rest] = values[first:]
            self.timestamps[:rest] = timestamps[first:]
        self._write = (self._write + n) % self.capacity
        self._count = min(self.capacity, self._count + n)

    def snapshot(self, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if self._count == 0:
            return (
                np.empty((0, self.channels), dtype=np.float32),
                np.empty(0, dtype=np.float64),
            )
        n = self._count if n is None else min(int(n), self._count)
        end = self._write
        start = (end - n) % self.capacity
        if start < end:
            values = self.values[start:end].copy()
            timestamps = self.timestamps[start:end].copy()
        else:
            values = np.concatenate((self.values[start:], self.values[:end]), axis=0)
            timestamps = np.concatenate((self.timestamps[start:], self.timestamps[:end]))
        return values, timestamps

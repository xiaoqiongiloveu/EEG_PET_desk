"""异步会话记录。

目录结构：
    sessions/20260304_153000/
        session.json     元数据与统计
        samples.bin      float32 交错采样 (n, 8)
        timestamps.bin   float64 主机时间戳 (n,)
        events.jsonl     事件流
        commands.log     下发的命令
        impedance.tsv    阻抗记录
        notes.txt        备注
"""

from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SAMPLE_DTYPE = np.dtype("<f4")
TIMESTAMP_DTYPE = np.dtype("<f8")


class SessionRecorder:
    """独立写盘线程 + 队列，避免阻塞采集与绘图。"""

    def __init__(self, root: Path, metadata: dict, channel_names: list[str]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = root / stamp
        suffix = 1
        while path.exists():
            path = root / f"{stamp}_{suffix:02d}"
            suffix += 1
        path.mkdir(parents=True)
        self.path = path
        self.channel_names = list(channel_names)
        self.sample_rate = int(metadata.get("sample_rate", 250))
        self.gain = int(metadata.get("gain", 24))
        self.started_at = time.time()
        # 相对时间以首个写入的数据（采样或阻抗）为 0 点，避免构造与首批数据之间的间隔造成偏移。
        self._time_origin: float | None = None

        self._samples_file = open(path / "samples.bin", "wb")
        self._timestamps_file = open(path / "timestamps.bin", "wb")
        self._events_file = open(path / "events.jsonl", "a", encoding="utf-8")
        self._commands_file = open(path / "commands.log", "a", encoding="utf-8")
        self._impedance_file = open(path / "impedance.tsv", "w", encoding="utf-8", newline="")
        self._impedance_file.write("time_s\t" + "\t".join(self.channel_names) + "\n")
        (path / "notes.txt").write_text("", encoding="utf-8")

        self._meta = {
            "created_at": datetime.fromtimestamp(self.started_at).isoformat(timespec="seconds"),
            "sample_rate": self.sample_rate,
            "gain": self.gain,
            "channel_names": self.channel_names,
            "channel_order": "raw",
            **metadata,
        }
        self._queue: queue.Queue = queue.Queue(maxsize=2048)
        self._sample_count = 0
        self._impedance_count = 0
        self._closed = False
        self._error = ""
        self._thread = threading.Thread(target=self._writer_loop, name="SessionWriter", daemon=True)
        self._thread.start()
        self.log_event("record_start", "开始记录", path=str(path))

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started_at

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def error(self) -> str:
        return self._error

    def append(self, values: np.ndarray, timestamps: np.ndarray) -> None:
        if self._closed or len(values) == 0:
            return
        try:
            relative = np.asarray(timestamps, dtype=np.float64)
            if relative.size:
                relative = relative - self._time_origin_of(relative[0])
            self._queue.put_nowait(
                (
                    "samples",
                    np.ascontiguousarray(values, dtype=SAMPLE_DTYPE),
                    np.ascontiguousarray(relative, dtype=TIMESTAMP_DTYPE),
                )
            )
        except queue.Full:
            self._error = "记录队列已满，部分采样未写入"

    def append_impedance(self, values: np.ndarray, timestamp: float) -> None:
        if self._closed:
            return
        relative = float(timestamp) - self._time_origin_of(float(timestamp))
        self._queue.put(("impedance", np.asarray(values, dtype=np.float64).copy(), relative))

    def _time_origin_of(self, timestamp: float) -> float:
        """记录第一个数据点的时间，作为会话相对时间 0 点。"""
        if self._time_origin is None:
            self._time_origin = float(timestamp)
        return self._time_origin

    def log_event(self, kind: str, message: str = "", **extra) -> None:
        if self._closed:
            return
        record = {
            "time": round(self.elapsed_s, 4),
            "epoch": time.time(),
            "kind": kind,
            "message": message,
            **extra,
        }
        self._queue.put(("event", record))

    def log_command(self, text: str) -> None:
        if self._closed:
            return
        self._queue.put(("command", f"{self.elapsed_s:9.3f}\t{text}"))

    def stop(self) -> Path:
        if self._closed:
            return self.path
        self.log_event("record_stop", "结束记录")
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        self._closed = True
        for handle in (
            self._samples_file,
            self._timestamps_file,
            self._events_file,
            self._commands_file,
            self._impedance_file,
        ):
            try:
                handle.close()
            except OSError:
                pass
        self._meta.update(
            {
                "ended_at": datetime.now().isoformat(timespec="seconds"),
                "duration_s": round(time.time() - self.started_at, 3),
                "sample_count": self._sample_count,
                "impedance_count": self._impedance_count,
                "error": self._error,
            }
        )
        (self.path / "session.json").write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.path

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                self._write_item(item)
            except OSError as exc:
                self._error = f"写盘失败: {exc}"
            finally:
                self._queue.task_done()

    def _write_item(self, item: tuple) -> None:
        kind = item[0]
        if kind == "samples":
            values, timestamps = item[1], item[2]
            self._samples_file.write(values.tobytes())
            self._timestamps_file.write(timestamps.tobytes())
            self._sample_count += int(values.shape[0])
        elif kind == "impedance":
            values, timestamp = item[1], item[2]
            row = "\t".join(f"{value:.1f}" for value in values)
            self._impedance_file.write(f"{timestamp:.3f}\t{row}\n")
            self._impedance_count += 1
        elif kind == "event":
            self._events_file.write(json.dumps(item[1], ensure_ascii=False) + "\n")
            self._events_file.flush()
        elif kind == "command":
            self._commands_file.write(item[1] + "\n")
            self._commands_file.flush()

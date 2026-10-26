"""无硬件时的模拟数据源。

生成的字节流与真实设备使用同一套 55AA 帧格式，并经过同一个 FrameParser 解析，
因此界面与算法链路在模拟模式下与实机完全一致。
"""

from __future__ import annotations

import time

import numpy as np

from ..constants import (
    CHANNELS_RAW,
    CMD_GET_STATE,
    CMD_START_STREAM,
    CMD_STOP_STREAM,
    CMD_TEST_IMPEDANCE,
    TYPE_AUX,
    TYPE_EEG,
    TYPE_IMPEDANCE,
    TYPE_STATUS,
)
from ..core.models import AcquisitionStats, SampleBatch
from ..protocol import (
    FrameParser,
    pack_frame,
    parse_aux_payload,
    parse_eeg_payload,
    parse_status_payload,
)
from .base import AcquisitionWorker, StreamConfig

CHANNEL_COUNT = len(CHANNELS_RAW)


class SimulatorWorker(AcquisitionWorker):
    """250 Hz 合成 EEG，含 alpha 节律、工频干扰、眨眼与运动伪迹。"""

    def __init__(self, config: StreamConfig | None = None, parent=None) -> None:
        super().__init__(parent)
        self.config = config or StreamConfig()
        self._stop_requested = False
        self._streaming = False
        self._impedance_enabled = False
        self._parser = FrameParser()
        self._sequence = 0
        self._sample_index = 0
        self._frames = 0
        self._samples = 0
        self._start_time = 0.0
        self._last_stats_time = 0.0
        self._last_stats_frames = 0
        self._last_stats_samples = 0

        self._rng = np.random.default_rng(20260304)
        self._phase = self._rng.uniform(0, 2 * np.pi, CHANNEL_COUNT)
        self._alpha_gain = self._rng.uniform(0.6, 1.4, CHANNEL_COUNT)
        self._noise_state = np.zeros(CHANNEL_COUNT, dtype=np.float64)
        self._batch_values: list[np.ndarray] = []
        self._batch_lead_off: list[np.ndarray] = []
        self._batch_timestamps: list[float] = []
        self._batch_device_ms: list[int] = []
        self._batch_sequences: list[int] = []
        self._batch_started = 0.0
        self._blink_until = 0.0
        self._motion_until = 0.0
        self._next_blink = 6.0
        self._next_motion = 15.0

    # ------------------------------------------------------------------ 控制
    def request_stop(self) -> None:
        self._stop_requested = True

    def start_stream(self) -> None:
        self._streaming = True
        self.log_message.emit("模拟数据流已启动")

    def stop_stream(self) -> None:
        self._streaming = False
        self.log_message.emit("模拟数据流已停止")

    def start_impedance(self) -> None:
        if not self._streaming:
            self._streaming = True
        self._impedance_enabled = True
        self.log_message.emit("模拟阻抗测试已开始")

    def stop_impedance(self) -> None:
        self._impedance_enabled = False
        self.log_message.emit("模拟阻抗测试已结束")

    def query_state(self) -> None:
        self._emit_status()

    def send_command(self, command_id: int, payload: bytes = b"") -> None:
        if command_id == CMD_START_STREAM:
            self.start_stream()
        elif command_id == CMD_STOP_STREAM:
            self.stop_stream()
        elif command_id == CMD_TEST_IMPEDANCE:
            self.start_impedance()
        elif command_id == CMD_GET_STATE:
            self._emit_status()

    # ------------------------------------------------------------------ 线程
    def run(self) -> None:
        self.connected.emit(True, "SIMULATOR")
        self.device_ready.emit("SIMULATOR")
        self.log_message.emit("已进入模拟数据模式（无需硬件）")
        self._streaming = bool(self.config.auto_start)
        self._start_time = time.monotonic()
        self._last_stats_time = self._start_time
        next_sample_time = time.monotonic()
        next_aux_time = next_sample_time
        next_status_time = next_sample_time
        next_imp_time = next_sample_time

        try:
            while not self._stop_requested:
                now = time.monotonic()
                if self._streaming:
                    period = 1.0 / max(self.config.sample_rate, 1)
                    while next_sample_time <= now:
                        self._emit_sample(next_sample_time)
                        next_sample_time += period
                else:
                    next_sample_time = max(next_sample_time, now)

                if now >= next_aux_time:
                    self._emit_aux()
                    next_aux_time = now + 0.5
                if now >= next_status_time:
                    self._emit_status()
                    next_status_time = now + 2.0
                if self._impedance_enabled and now >= next_imp_time:
                    self._emit_impedance()
                    next_imp_time = now + 1.0

                self._flush_batch(now, force=False)
                if now - self._last_stats_time >= 1.0:
                    self._emit_stats(now)
                time.sleep(0.005)
        finally:
            self._flush_batch(time.monotonic(), force=True)
            self._emit_stats(time.monotonic())
            self.connected.emit(False, "")

    # ------------------------------------------------------------------ 合成
    def _emit_sample(self, host_time: float) -> None:
        fs = max(self.config.sample_rate, 1)
        t = self._sample_index / fs
        elapsed = host_time - self._start_time

        if elapsed >= self._next_blink:
            self._blink_until = elapsed + 0.4
            self._next_blink = elapsed + self._rng.uniform(6.0, 12.0)
        if elapsed >= self._next_motion:
            self._motion_until = elapsed + 1.5
            self._next_motion = elapsed + self._rng.uniform(20.0, 35.0)

        # 1/f 噪声：一阶低通白噪声，比纯白噪声更接近真实脑电背景。
        white = self._rng.normal(0.0, 2.2, CHANNEL_COUNT)
        self._noise_state = 0.92 * self._noise_state + 0.08 * white * 10.0

        alpha = 14.0 * self._alpha_gain * np.sin(2 * np.pi * 10.0 * t + self._phase)
        line = 4.0 * np.sin(2 * np.pi * 50.0 * t)
        values = self._noise_state + alpha + line

        if elapsed < self._blink_until:
            envelope = np.sin(np.pi * (self._blink_until - elapsed) / 0.4)
            frontal = np.array([0.3, 0.3, 0.4, 0.3, 0.3, 1.0, 0.9, 1.0])
            values = values + 90.0 * envelope * frontal
        if elapsed < self._motion_until:
            values = values + self._rng.normal(0.0, 25.0, CHANNEL_COUNT)

        raw = np.rint(values * (self.config.gain * 8_388_607.0) / 4_500_000.0)
        raw = np.clip(raw, -8_388_608, 8_388_607).astype("<i4")
        payload = bytearray(47)
        payload[3:35] = raw.tobytes()
        acc = np.array(
            [0.0, 0.0, 1.0 if elapsed >= self._motion_until else 1.3], dtype="<i2"
        )
        payload[35:41] = (acc * 1000).astype("<i2").tobytes()

        self._sequence = (self._sequence + 1) & 0xFFFF
        device_ms = int(t * 1000.0) & 0xFFFFFFFF
        frame_bytes = pack_frame(TYPE_EEG, self._sequence, device_ms, bytes(payload))
        self._sample_index += 1

        for frame in self._parser.feed(frame_bytes):
            if frame.packet_type != TYPE_EEG:
                continue
            self._frames += 1
            self._samples += 1
            parsed = parse_eeg_payload(frame.payload, self.config.gain)
            if not self._batch_values:
                self._batch_started = host_time
            self._batch_values.append(parsed.values_uv)
            self._batch_lead_off.append(parsed.lead_off)
            self._batch_timestamps.append(host_time)
            self._batch_device_ms.append(frame.device_timestamp_ms)
            self._batch_sequences.append(frame.sequence)

    def _emit_impedance(self) -> None:
        base = np.array([4.2, 3.8, 5.6, 4.9, 6.1, 3.5, 7.2, 4.4])
        jitter = self._rng.normal(0.0, 0.6, CHANNEL_COUNT)
        values = np.clip(base + jitter, 0.5, 200.0).astype(np.float32)
        payload = np.rint(values * 1000).astype("<u2").tobytes()
        for frame in self._parser.feed(pack_frame(TYPE_IMPEDANCE, 0, 0, payload)):
            self.impedance_received.emit(
                np.frombuffer(frame.payload, dtype="<u2", count=8).astype(np.float32)
            )

    def _emit_aux(self) -> None:
        payload = bytearray(16)
        acc = np.array([0.01, 0.02, 1.0]) * 1000
        payload[0:6] = np.rint(acc).astype("<i2")[:3].tobytes()
        payload[6:12] = np.zeros(3, dtype="<i2").tobytes()
        payload[12:14] = int(3.92 * 100).to_bytes(2, "little")
        elapsed = time.monotonic() - self._start_time
        payload[14] = 45 if elapsed < self._motion_until else 0
        for frame in self._parser.feed(pack_frame(TYPE_AUX, 0, 0, bytes(payload))):
            self.aux_received.emit(parse_aux_payload(frame.payload))

    def _emit_status(self) -> None:
        payload = bytearray(24)
        payload[0] = 1
        payload[1] = 12
        payload[2] = 1
        payload[3:5] = (392).to_bytes(2, "little")
        payload[5] = 3 if self._streaming else 1
        payload[11:19] = b"SIM-1.0\x00"
        payload[19:24] = b"SIM01"
        for frame in self._parser.feed(pack_frame(TYPE_STATUS, 0, 0, bytes(payload))):
            self.status_received.emit(parse_status_payload(frame.payload))

    def _flush_batch(self, now: float, force: bool) -> None:
        if not self._batch_values:
            return
        if not force and len(self._batch_values) < 10 and now - self._batch_started < 0.03:
            return
        batch = SampleBatch(
            values=np.vstack(self._batch_values).astype(np.float32, copy=False),
            timestamps=np.asarray(self._batch_timestamps, dtype=np.float64),
            device_ms=np.asarray(self._batch_device_ms, dtype=np.uint32),
            lead_off=np.vstack(self._batch_lead_off),
            sequences=np.asarray(self._batch_sequences, dtype=np.uint16),
        )
        self._batch_values.clear()
        self._batch_lead_off.clear()
        self._batch_timestamps.clear()
        self._batch_device_ms.clear()
        self._batch_sequences.clear()
        self.samples_batch.emit(batch)

    def _emit_stats(self, now: float) -> None:
        elapsed = max(now - self._last_stats_time, 1e-6)
        stats = AcquisitionStats(
            frames=self._frames,
            samples=self._samples,
            crc_errors=self._parser.crc_errors,
            malformed_frames=self._parser.malformed_frames,
            frame_rate=(self._frames - self._last_stats_frames) / elapsed,
            sample_rate=(self._samples - self._last_stats_samples) / elapsed,
            elapsed_s=now - self._start_time,
            packet_counts={TYPE_EEG: self._frames},
        )
        self._last_stats_time = now
        self._last_stats_frames = self._frames
        self._last_stats_samples = self._samples
        self.stats_received.emit(stats)

"""串口采集线程：建立连接、解析数据、下发命令。"""

from __future__ import annotations

import queue
import time

import numpy as np
import serial

from ..constants import (
    CMD_DISABLE_DEVICE,
    CMD_ENABLE_DEVICE,
    CMD_GET_STATE,
    CMD_SET_CONFIG,
    CMD_START_STREAM,
    CMD_STOP_IMPEDANCE,
    CMD_STOP_STREAM,
    CMD_TEST_IMPEDANCE,
    COMMAND_NAMES,
    TYPE_AUX,
    TYPE_COMMAND_RESPONSE,
    TYPE_EEG,
    TYPE_IMPEDANCE,
    TYPE_STATUS,
)
from ..core.models import AcquisitionStats, SampleBatch
from ..protocol import (
    FrameParser,
    build_command_frame,
    build_config_payload,
    parse_aux_payload,
    parse_command_response,
    parse_eeg_payload,
    parse_impedance_payload,
    parse_status_payload,
)
from .base import AcquisitionWorker, StreamConfig
from .port_scan import resolve_port

# 命令之间的最小发送间隔，避免连续写把固件命令队列打满。
COMMAND_GAP_S = 0.12
SLOW_COMMAND_GAP_S = 0.35
SLOW_COMMANDS = (CMD_STOP_STREAM, CMD_STOP_IMPEDANCE, CMD_DISABLE_DEVICE)

# 打开串口后等待设备有效响应的时间：WiFi 透传需考虑模块配对，给更长的窗口。
USB_READY_TIMEOUT_S = 3.0
WIFI_READY_TIMEOUT_S = 6.0


class SerialWorker(AcquisitionWorker):
    """独占一个串口，读取 55AA 帧并批量向上层推送。"""

    def __init__(self, config: StreamConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.port = ""
        self._serial: serial.Serial | None = None
        self._parser = FrameParser()
        self._commands: queue.Queue[tuple[int, bytes]] = queue.Queue()
        self._stop_requested = False
        self._device_ready = False
        self._ready_deadline = 0.0

        self._command_sequence = 0
        self._next_send_time = 0.0

        self._last_sequence: int | None = None
        self._sequence_gaps = 0
        self._dropped_samples = 0
        self._frames = 0
        self._samples = 0
        self._bytes_received = 0
        self._packet_counts: dict[int, int] = {}
        self._start_time = 0.0
        self._last_stats_time = 0.0
        self._last_stats_frames = 0
        self._last_stats_samples = 0

        self._clock_host: float | None = None
        self._clock_elapsed_ms = 0.0
        self._clock_last_device_ms: int | None = None

        self._batch_values: list[np.ndarray] = []
        self._batch_lead_off: list[np.ndarray] = []
        self._batch_timestamps: list[float] = []
        self._batch_device_ms: list[int] = []
        self._batch_sequences: list[int] = []
        self._batch_started = 0.0

    # ------------------------------------------------------------------ 控制
    def request_stop(self) -> None:
        self._stop_requested = True

    def start_stream(self) -> None:
        self._commands.put((CMD_START_STREAM, b""))

    def stop_stream(self) -> None:
        self._commands.put((CMD_STOP_STREAM, b""))

    def start_impedance(self) -> None:
        self._commands.put((CMD_TEST_IMPEDANCE, b""))
        self.log_message.emit("已请求阻抗测试，等待设备返回数据")

    def stop_impedance(self) -> None:
        self._commands.put((CMD_STOP_IMPEDANCE, b""))

    def query_state(self) -> None:
        self._commands.put((CMD_GET_STATE, b""))

    def send_command(self, command_id: int, payload: bytes = b"") -> None:
        self._commands.put((int(command_id) & 0xFF, bytes(payload)))

    def apply_config(self, sample_rate: int, gain: int) -> None:
        self.config.sample_rate = int(sample_rate)
        self.config.gain = int(gain)
        payload = build_config_payload(sample_rate, gain)
        self._commands.put((CMD_SET_CONFIG, payload))

    # ------------------------------------------------------------------ 线程
    def run(self) -> None:  # noqa: C901 - 采集主循环
        port = resolve_port(self.config.port, strict=self.config.wifi)
        if not port:
            if self.config.wifi:
                self.error_occurred.emit(
                    f"未找到 WiFi 模块串口 {self.config.port or '（未选择）'}，"
                    "请插入模块后点击刷新并手动选择对应 COM 口"
                )
            else:
                self.error_occurred.emit("未找到可用串口，请检查设备连接")
            self.connected.emit(False, "")
            return
        self.port = port
        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=int(self.config.baudrate),
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.02,
                write_timeout=0.5,
            )
            self._serial.reset_input_buffer()
        except (serial.SerialException, OSError) as exc:
            self.error_occurred.emit(f"打开串口 {port} 失败: {exc}")
            self.connected.emit(False, port)
            return

        self.connected.emit(True, port)
        self._ready_deadline = time.monotonic() + (
            WIFI_READY_TIMEOUT_S if self.config.wifi else USB_READY_TIMEOUT_S
        )
        mode = "WiFi 透传" if self.config.wifi else "USB"
        self.log_message.emit(
            f"已打开 {port} @ {self.config.baudrate} bps（{mode}，8N1），等待设备响应"
        )
        self._start_time = time.monotonic()
        self._last_stats_time = self._start_time

        try:
            self._bootstrap()
            while not self._stop_requested:
                now = time.monotonic()
                self._drain_commands(now)
                waiting = self._serial.in_waiting
                data = self._serial.read(waiting if waiting > 0 else 1)
                if data:
                    self._bytes_received += len(data)
                    self._handle_frames(self._parser.feed(data))
                self._flush_batch(now, force=False)
                if not self._device_ready and now >= self._ready_deadline:
                    self.error_occurred.emit(self._timeout_message(port))
                    break
                if now - self._last_stats_time >= 1.0:
                    self._emit_stats(now)
        except (serial.SerialException, OSError) as exc:
            if not self._stop_requested:
                self.error_occurred.emit(f"串口读写异常: {exc}")
        finally:
            self._flush_batch(time.monotonic(), force=True)
            self._emit_stats(time.monotonic())
            self._send_stop_best_effort()
            self._parser.buffer.clear()
            self._close_serial()
            self.connected.emit(False, "")

    def _timeout_message(self, port: str) -> str:
        if self.config.wifi:
            return (
                f"{port} 已打开，但 {WIFI_READY_TIMEOUT_S:.0f} 秒内未收到设备有效数据。"
                f"请确认 WiFi 模块已自动配对、波特率是否为 {self.config.baudrate}，"
                "并核对设备是否已上电。"
            )
        return (
            f"{port} 已打开，但 {USB_READY_TIMEOUT_S:.0f} 秒内未收到设备有效数据，"
            "请检查 USB 连接或设备供电。"
        )

    def _bootstrap(self) -> None:
        """上电初始化：使能 → 配置 → 查询状态 → 开始采集。"""

        self._write_command(CMD_ENABLE_DEVICE, b"")
        time.sleep(0.10)
        self._write_command(
            CMD_SET_CONFIG,
            build_config_payload(self.config.sample_rate, self.config.gain),
        )
        time.sleep(0.10)
        self._write_command(CMD_GET_STATE, b"")
        time.sleep(0.05)
        if self.config.auto_start:
            self._write_command(CMD_START_STREAM, b"")
            time.sleep(0.05)
            self._next_send_time = time.monotonic() + COMMAND_GAP_S
        self.log_message.emit(
            f"设备配置完成：{self.config.sample_rate} Hz，增益 {self.config.gain}"
        )

    def _drain_commands(self, now: float) -> None:
        if now < self._next_send_time or self._serial is None:
            return
        try:
            command_id, payload = self._commands.get_nowait()
        except queue.Empty:
            return
        self._write_command(command_id, payload)
        gap = SLOW_COMMAND_GAP_S if command_id in SLOW_COMMANDS else COMMAND_GAP_S
        self._next_send_time = now + gap

    def _write_command(self, command_id: int, payload: bytes = b"") -> None:
        if self._serial is None:
            return
        self._command_sequence = (self._command_sequence + 1) & 0xFFFF
        frame = build_command_frame(command_id, self._command_sequence, payload)
        self._serial.write(frame)
        self._serial.flush()
        name = COMMAND_NAMES.get(command_id, f"0x{command_id:02X}")
        self.log_message.emit(f"→ {name}")

    # ------------------------------------------------------------------ 数据
    def _handle_frames(self, frames) -> None:
        if frames and not self._device_ready:
            self._device_ready = True
            self.log_message.emit("收到设备有效帧，链路已建立")
            self.device_ready.emit(self.port)
        for frame in frames:
            self._frames += 1
            self._packet_counts[frame.packet_type] = (
                self._packet_counts.get(frame.packet_type, 0) + 1
            )
            if frame.packet_type == TYPE_EEG:
                self._handle_eeg(frame)
            elif frame.packet_type == TYPE_IMPEDANCE:
                try:
                    self.impedance_received.emit(parse_impedance_payload(frame.payload))
                except ValueError as exc:
                    self.log_message.emit(f"阻抗数据异常: {exc}")
            elif frame.packet_type == TYPE_STATUS:
                try:
                    self.status_received.emit(parse_status_payload(frame.payload))
                except ValueError as exc:
                    self.log_message.emit(f"状态数据异常: {exc}")
            elif frame.packet_type == TYPE_AUX:
                try:
                    self.aux_received.emit(parse_aux_payload(frame.payload))
                except ValueError as exc:
                    self.log_message.emit(f"辅助数据异常: {exc}")
            elif frame.packet_type == TYPE_COMMAND_RESPONSE:
                try:
                    response = parse_command_response(frame.payload)
                except ValueError as exc:
                    self.log_message.emit(f"命令响应异常: {exc}")
                else:
                    self.command_response.emit(response)
                    name = COMMAND_NAMES.get(response.command_id, f"0x{response.command_id:02X}")
                    result = "成功" if response.ok else f"失败({response.status})"
                    detail = f" {response.message}" if response.message else ""
                    self.log_message.emit(f"← {name} {result}{detail}")

    def _handle_eeg(self, frame) -> None:
        try:
            parsed = parse_eeg_payload(frame.payload, self.config.gain)
        except ValueError as exc:
            self.log_message.emit(f"EEG 数据异常: {exc}")
            return
        self._samples += 1
        if self._last_sequence is not None:
            gap = (frame.sequence - self._last_sequence - 1) & 0xFFFF
            if 0 < gap < 1000:
                self._sequence_gaps += 1
                self._dropped_samples += gap
        self._last_sequence = frame.sequence

        timestamp = self._device_timestamp(frame.device_timestamp_ms)
        if not self._batch_values:
            self._batch_started = time.monotonic()
        self._batch_values.append(parsed.values_uv)
        self._batch_lead_off.append(parsed.lead_off)
        self._batch_timestamps.append(timestamp)
        self._batch_device_ms.append(frame.device_timestamp_ms)
        self._batch_sequences.append(frame.sequence)

    def _device_timestamp(self, device_ms: int) -> float:
        """把 32 位设备时间戳换算成主机单调时钟秒，并处理回绕。"""

        if self._clock_host is None:
            self._clock_host = time.monotonic()
            self._clock_elapsed_ms = 0.0
            self._clock_last_device_ms = device_ms
            return self._clock_host
        assert self._clock_last_device_ms is not None
        delta = (device_ms - self._clock_last_device_ms) & 0xFFFFFFFF
        if delta >= 0x80000000:
            delta -= 0x100000000
        self._clock_elapsed_ms += delta
        self._clock_last_device_ms = device_ms
        return self._clock_host + self._clock_elapsed_ms / 1000.0

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
            bytes_received=self._bytes_received,
            crc_errors=self._parser.crc_errors,
            malformed_frames=self._parser.malformed_frames,
            sequence_gaps=self._sequence_gaps,
            dropped_samples=self._dropped_samples,
            frame_rate=(self._frames - self._last_stats_frames) / elapsed,
            sample_rate=(self._samples - self._last_stats_samples) / elapsed,
            elapsed_s=now - self._start_time,
            packet_counts=dict(self._packet_counts),
        )
        self._last_stats_time = now
        self._last_stats_frames = self._frames
        self._last_stats_samples = self._samples
        self.stats_received.emit(stats)

    def _send_stop_best_effort(self) -> None:
        """断开前尽量让设备停止采集。"""

        if self._serial is None:
            return
        try:
            self._write_command(CMD_STOP_STREAM, b"")
            time.sleep(SLOW_COMMAND_GAP_S)
        except (serial.SerialException, OSError):
            pass

    def _close_serial(self) -> None:
        if self._serial is None:
            return
        try:
            self._serial.close()
        except (serial.SerialException, OSError):
            pass
        self._serial = None

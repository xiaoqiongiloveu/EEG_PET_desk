"""采集控制器：统一管理数据源、实时滤波与记录。"""

from __future__ import annotations

import time

from PySide6.QtCore import QObject, Signal

from .config import Settings
from .constants import CHANNELS_RAW, WIFI_SAMPLE_RATE
from .core.filters import RealtimeFilter, apply_rereference
from .core.models import SampleBatch
from .storage import SessionRecorder
from .transport import SerialWorker, SimulatorWorker, StreamConfig
from .transport.base import AcquisitionWorker


class AcquisitionController(QObject):
    samples_ready = Signal(object)  # 已滤波的 SampleBatch
    connected_changed = Signal(bool, str)
    device_ready = Signal(str)  # 收到设备首帧有效数据
    stats_updated = Signal(object)
    impedance_updated = Signal(object)
    status_updated = Signal(object)
    aux_updated = Signal(object)
    command_response = Signal(object)
    log = Signal(str)
    recording_changed = Signal(bool, str)
    error = Signal(str)

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.channel_names = settings.normalized_channels()
        self.worker: AcquisitionWorker | None = None
        self.recorder: SessionRecorder | None = None
        self.source = "serial"
        self._mode = "usb"
        self._active_baudrate = settings.baudrate
        self._filter = RealtimeFilter(settings.sample_rate, len(CHANNELS_RAW))
        self._apply_filter_settings()
        self._reref_mode = settings.reref_mode
        self._reref_channel = settings.reref_channel
        self._impedance_epoch = time.monotonic()

    # ------------------------------------------------------------------ 连接
    @property
    def connected(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    @property
    def recording(self) -> bool:
        return self.recorder is not None

    @property
    def mode(self) -> str:
        """当前连接模式：usb / wifi / simulator。"""

        return self._mode

    @property
    def active_baudrate(self) -> int:
        return self._active_baudrate

    def connect_serial(
        self,
        port: str,
        baudrate: int,
        auto_start: bool = True,
        wifi: bool = False,
    ) -> None:
        self.disconnect_device()
        self.source = "serial"
        self._mode = "wifi" if wifi else "usb"
        self._active_baudrate = int(baudrate)
        if wifi and self.settings.sample_rate != WIFI_SAMPLE_RATE:
            # 界面已在切换模式时提示；这里再兜底保证滤波与设备配置一致。
            self.log.emit(f"WiFi 模式仅支持 {WIFI_SAMPLE_RATE} SPS，已按该采样率配置")
            self.settings.sample_rate = WIFI_SAMPLE_RATE
            self._filter = RealtimeFilter(self.settings.sample_rate, len(CHANNELS_RAW))
            self._apply_filter_settings()
        config = StreamConfig(
            port=port,
            baudrate=baudrate,
            sample_rate=WIFI_SAMPLE_RATE if wifi else self.settings.sample_rate,
            gain=self.settings.gain,
            auto_start=auto_start,
            wifi=wifi,
        )
        self.worker = SerialWorker(config, self)
        self._attach(self.worker)
        self.worker.start()

    def connect_simulator(self, auto_start: bool = True) -> None:
        self.disconnect_device()
        self.source = "simulator"
        self._mode = "simulator"
        config = StreamConfig(
            sample_rate=self.settings.sample_rate,
            gain=self.settings.gain,
            auto_start=auto_start,
        )
        self.worker = SimulatorWorker(config, self)
        self._attach(self.worker)
        self.worker.start()

    def disconnect_device(self) -> None:
        if self.worker is None:
            return
        self.worker.request_stop()
        if not self.worker.wait(2500):
            self.worker.terminate()
            self.worker.wait(500)
        self.worker.deleteLater()
        self.worker = None
        if self.recorder is not None:
            self.stop_recording()

    def _attach(self, worker: AcquisitionWorker) -> None:
        worker.connected.connect(self._on_connected)
        worker.device_ready.connect(self.device_ready)
        worker.samples_batch.connect(self._on_samples)
        worker.impedance_received.connect(self._on_impedance)
        worker.status_received.connect(self._on_status)
        worker.aux_received.connect(self._on_aux)
        worker.stats_received.connect(self._on_stats)
        worker.command_response.connect(self._on_response)
        worker.log_message.connect(self._on_log)
        worker.error_occurred.connect(self._on_error)

    # ------------------------------------------------------------------ 控制
    def start_stream(self) -> None:
        if self.worker is not None:
            self.worker.start_stream()

    def stop_stream(self) -> None:
        if self.worker is not None:
            self.worker.stop_stream()

    def start_impedance(self) -> None:
        self._impedance_epoch = time.monotonic()
        if self.worker is not None:
            self.worker.start_impedance()

    def stop_impedance(self) -> None:
        if self.worker is not None:
            self.worker.stop_impedance()

    def query_state(self) -> None:
        if self.worker is not None:
            self.worker.query_state()

    def send_command(self, command_id: int, payload: bytes = b"") -> None:
        if self.worker is not None:
            self.worker.send_command(command_id, payload)

    def apply_config(self, sample_rate: int, gain: int) -> None:
        if self._mode == "wifi" and int(sample_rate) != WIFI_SAMPLE_RATE:
            # WiFi 透传本版固定 250 SPS，避免把其他采样率下发给设备。
            self.log.emit(
                f"WiFi 模式仅支持 {WIFI_SAMPLE_RATE} SPS，忽略采样率 {sample_rate} Hz"
            )
            sample_rate = WIFI_SAMPLE_RATE
        self.settings.sample_rate = int(sample_rate)
        self.settings.gain = int(gain)
        self._filter = RealtimeFilter(sample_rate, len(CHANNELS_RAW))
        self._apply_filter_settings()
        if isinstance(self.worker, SerialWorker):
            self.worker.apply_config(sample_rate, gain)

    def set_rereference(self, mode: str, channel: str = "") -> None:
        self._reref_mode = mode
        self._reref_channel = channel

    def configure_filter(
        self,
        low: float,
        high: float,
        notch: int,
        enabled: bool,
        sample_rate: int | None = None,
    ) -> None:
        if sample_rate is not None:
            self.settings.sample_rate = int(sample_rate)
        self.settings.filter_low = float(low)
        self.settings.filter_high = float(high)
        self.settings.notch = int(notch)
        self.settings.filter_enabled = bool(enabled)
        self._filter.configure(
            self.settings.sample_rate,
            self.settings.filter_low,
            self.settings.filter_high,
            self.settings.notch,
            self.settings.filter_enabled,
        )
        self._filter.reset()

    def _apply_filter_settings(self) -> None:
        self._filter.configure(
            self.settings.sample_rate,
            self.settings.filter_low,
            self.settings.filter_high,
            self.settings.notch,
            self.settings.filter_enabled,
        )

    # ------------------------------------------------------------------ 记录
    def start_recording(self) -> str | None:
        if self.recorder is not None:
            return str(self.recorder.path)
        # 以工作线程的实际配置为准，避免设置页改值后与实际数据流不一致。
        worker_config = getattr(self.worker, "config", None)
        sample_rate = int(getattr(worker_config, "sample_rate", self.settings.sample_rate))
        gain = int(getattr(worker_config, "gain", self.settings.gain))
        metadata = {
            "source": self.source,
            "mode": self._mode,
            "device": getattr(self.worker, "port", "") if self.source == "serial" else "SIMULATOR",
            "baudrate": self._active_baudrate,
            "sample_rate": sample_rate,
            "gain": gain,
            "firmware": getattr(self, "_firmware", ""),
            "serial_no": getattr(self, "_serial_no", ""),
            "filter": {
                "enabled": self.settings.filter_enabled,
                "low": self.settings.filter_low,
                "high": self.settings.filter_high,
                "notch": self.settings.notch,
            },
            "reref": {"mode": self._reref_mode, "channel": self._reref_channel},
        }
        try:
            self.recorder = SessionRecorder(
                self.settings.resolved_storage_dir(),
                metadata,
                self.channel_names,
            )
        except OSError as exc:
            self.error.emit(f"创建记录文件失败: {exc}")
            return None
        self.log.emit(f"开始记录：{self.recorder.path}")
        self.recording_changed.emit(True, str(self.recorder.path))
        return str(self.recorder.path)

    def stop_recording(self) -> str | None:
        if self.recorder is None:
            return None
        path = self.recorder.stop()
        self.log.emit(f"记录已保存：{path}")
        self.recorder = None
        self.recording_changed.emit(False, str(path))
        return str(path)

    def add_marker(self, label: str) -> None:
        if self.recorder is not None:
            self.recorder.log_event("marker", label, label=label)
        self.log.emit(f"标记：{label}")

    def shutdown(self) -> None:
        self.stop_recording()
        self.disconnect_device()

    # ------------------------------------------------------------------ 回调
    def _on_connected(self, connected: bool, port: str) -> None:
        if connected:
            self._filter.reset()
        self.connected_changed.emit(connected, port)
        if not connected:
            self.log.emit("设备连接已断开")

    def _on_samples(self, batch: SampleBatch) -> None:
        if self.recorder is not None:
            self.recorder.append(batch.values, batch.timestamps)
        values = apply_rereference(
            batch.values,
            self.channel_names,
            self._reref_mode,
            self._reref_channel,
        )
        filtered = self._filter.process(values)
        out = SampleBatch(
            values=filtered,
            timestamps=batch.timestamps,
            device_ms=batch.device_ms,
            lead_off=batch.lead_off,
            sequences=batch.sequences,
        )
        self.samples_ready.emit(out)

    def _on_stats(self, stats) -> None:
        if self.recorder is not None:
            self.recorder.append_stats(stats)
        self.stats_updated.emit(stats)

    def _on_impedance(self, values) -> None:
        self.impedance_updated.emit(values)
        if self.recorder is not None:
            self.recorder.append_impedance(values, time.monotonic())

    def _on_status(self, status) -> None:
        self._firmware = status.firmware
        self._serial_no = status.serial_no
        self.status_updated.emit(status)

    def _on_aux(self, aux) -> None:
        self.aux_updated.emit(aux)

    def _on_response(self, response) -> None:
        self.command_response.emit(response)
        if self.recorder is not None:
            name = f"0x{response.command_id:02X}"
            self.recorder.log_event(
                "command_response",
                response.message,
                command=name,
                status=response.status,
            )

    def _on_log(self, message: str) -> None:
        self.log.emit(message)
        if self.recorder is not None and message.startswith("→"):
            self.recorder.log_command(message)

    def _on_error(self, message: str) -> None:
        self.error.emit(message)
        self.log.emit(f"错误：{message}")

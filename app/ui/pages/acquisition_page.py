"""实时采集页面。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QTabWidget, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel,
    CheckBox,
    ComboBox,
    DoubleSpinBox,
    FluentIcon,
    PrimaryPushButton,
    PushButton,
    SwitchButton,
    TransparentToolButton,
)

from ...analysis.quality import evaluate_channels
from ...config import Settings
from ...constants import CHANNELS_RAW, WIFI_SAMPLE_RATE
from ...controller import AcquisitionController
from ...core.models import SampleBatch
from ...protocol import DeviceStatus
from ...transport.port_scan import list_serial_ports
from ..notify import notify
from ..widgets.panels import ImpedancePanel, QualityTable, StatusPanel
from ..widgets.waveform import LiveWaveformWidget

NOTCH_OPTIONS = (("关闭", 0), ("50 Hz", 50), ("60 Hz", 60))
REREF_OPTIONS = (("设备参考", "none"), ("平均参考 (CAR)", "car"), ("指定通道", "channel"))


class AcquisitionPage(QWidget):
    def __init__(
        self,
        controller: AcquisitionController,
        settings: Settings,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("acquisitionPage")
        self.controller = controller
        self.settings = settings
        self._marker_count = 0
        self._stream_running = False
        self._impedance_running = False
        self._loading_ports = False
        self._build_ui()
        self._load_ports()
        self._load_settings()
        self._connect_signals()
        self._update_controls()

        self._quality_timer = QTimer(self)
        self._quality_timer.timeout.connect(self._update_quality)
        self._quality_timer.start(3000)

    # ------------------------------------------------------------------ 界面
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        toolbar.addWidget(CaptionLabel("串口", self))
        self.portCombo = ComboBox(self)
        self.portCombo.setMinimumWidth(170)
        toolbar.addWidget(self.portCombo)
        self.refreshButton = TransparentToolButton(FluentIcon.SYNC, self)
        self.refreshButton.setToolTip("刷新串口列表")
        toolbar.addWidget(self.refreshButton)
        self.wifiCheck = CheckBox("使用 WiFi 模块", self)
        self.wifiCheck.setToolTip("勾选后通过电脑端 WiFi 模块的虚拟串口透传，需手动选择对应 COM 口")
        toolbar.addWidget(self.wifiCheck)
        self.modeHint = CaptionLabel("", self)
        self.modeHint.setStyleSheet("color: #b26a00;")
        toolbar.addWidget(self.modeHint)
        self.connectButton = PrimaryPushButton("连接", self)
        self.simButton = PushButton("模拟数据", self)
        self.streamButton = PrimaryPushButton("开始采集", self)
        self.impedanceButton = PushButton("阻抗测试", self)
        self.recordButton = PushButton("开始记录", self)
        self.markerButton = PushButton("打标记", self)
        for widget in (
            self.connectButton,
            self.simButton,
            self.streamButton,
            self.impedanceButton,
            self.recordButton,
            self.markerButton,
        ):
            toolbar.addWidget(widget)
        toolbar.addStretch(1)
        self.rateLabel = CaptionLabel("-- Hz", self)
        toolbar.addWidget(self.rateLabel)
        layout.addLayout(toolbar)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        controls.addWidget(CaptionLabel("窗口", self))
        self.windowSpin = DoubleSpinBox(self)
        self.windowSpin.setRange(2.0, 60.0)
        self.windowSpin.setSingleStep(1.0)
        self.windowSpin.setSuffix(" s")
        controls.addWidget(self.windowSpin)

        controls.addWidget(CaptionLabel("幅值", self))
        self.scaleSpin = DoubleSpinBox(self)
        self.scaleSpin.setRange(10.0, 2000.0)
        self.scaleSpin.setSingleStep(10.0)
        self.scaleSpin.setSuffix(" µV")
        controls.addWidget(self.scaleSpin)

        self.filterSwitch = SwitchButton(self)
        controls.addWidget(CaptionLabel("显示滤波", self))
        controls.addWidget(self.filterSwitch)

        controls.addWidget(CaptionLabel("高通", self))
        self.lowSpin = DoubleSpinBox(self)
        self.lowSpin.setRange(0.1, 20.0)
        self.lowSpin.setSingleStep(0.1)
        self.lowSpin.setSuffix(" Hz")
        controls.addWidget(self.lowSpin)

        controls.addWidget(CaptionLabel("低通", self))
        self.highSpin = DoubleSpinBox(self)
        self.highSpin.setRange(10.0, 120.0)
        self.highSpin.setSingleStep(1.0)
        self.highSpin.setSuffix(" Hz")
        controls.addWidget(self.highSpin)

        controls.addWidget(CaptionLabel("陷波", self))
        self.notchCombo = ComboBox(self)
        for text, value in NOTCH_OPTIONS:
            self.notchCombo.addItem(text, userData=value)
        controls.addWidget(self.notchCombo)

        controls.addWidget(CaptionLabel("重参考", self))
        self.rerefCombo = ComboBox(self)
        for text, value in REREF_OPTIONS:
            self.rerefCombo.addItem(text, userData=value)
        controls.addWidget(self.rerefCombo)

        self.referenceCombo = ComboBox(self)
        self.referenceCombo.addItems(list(CHANNELS_RAW))
        self.referenceCombo.setEnabled(False)
        controls.addWidget(self.referenceCombo)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.waveform = LiveWaveformWidget(250, list(CHANNELS_RAW), parent=self)
        layout.addWidget(self.waveform, 1)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self.statusPanel = StatusPanel(self)
        self.statusPanel.setFixedWidth(360)
        self.qualityTable = QualityTable(self)
        self.impedancePanel = ImpedancePanel(list(CHANNELS_RAW), list(self.settings.normalized_order()), self)
        self.tabs = QTabWidget(self)
        self.tabs.addTab(self.impedancePanel, "电极阻抗")
        self.tabs.addTab(self.qualityTable, "信号质量")
        bottom.addWidget(self.statusPanel)
        bottom.addWidget(self.tabs, 1)
        layout.addLayout(bottom)

    def _load_ports(self) -> None:
        current = self.portCombo.currentData() if self.portCombo.count() else ""
        self._loading_ports = True
        try:
            self.portCombo.clear()
            ports = list_serial_ports()
            for port in ports:
                label = f"{port.device}  {port.description}".strip()
                if port.is_target:
                    label += "  [设备]"
                self.portCombo.addItem(label, userData=port.device)
            if not ports:
                self.portCombo.addItem("未发现串口", userData="")
        finally:
            self._loading_ports = False
        if current:
            self._select_port(current)

    def _select_port(self, port_name: str) -> None:
        if port_name:
            for index in range(self.portCombo.count()):
                if self.portCombo.itemData(index) == port_name:
                    self.portCombo.setCurrentIndex(index)
                    return
        if self.settings.wifi_mode:
            # WiFi 模块不按名称/VID 猜测，保留当前选择并由用户手动确认。
            return
        for index in range(self.portCombo.count()):
            if "设备" in self.portCombo.itemText(index):
                self.portCombo.setCurrentIndex(index)
                return
        if self.portCombo.count():
            self.portCombo.setCurrentIndex(0)

    def _load_settings(self) -> None:
        self.wifiCheck.blockSignals(True)
        self.wifiCheck.setChecked(bool(self.settings.wifi_mode))
        self.wifiCheck.blockSignals(False)
        self._update_mode_hint()
        self._select_port(self.settings.port)
        self.windowSpin.setValue(self.settings.window_s)
        self.scaleSpin.setValue(self.settings.scale_uv)
        self.filterSwitch.setChecked(self.settings.filter_enabled)
        self.lowSpin.setValue(self.settings.filter_low)
        self.highSpin.setValue(self.settings.filter_high)
        self._select_combo_data(self.notchCombo, self.settings.notch)
        self._select_combo_data(self.rerefCombo, self.settings.reref_mode)
        if self.settings.reref_channel:
            self._select_combo_text(self.referenceCombo, self.settings.reref_channel)
        self.waveform.set_window(self.settings.window_s)
        self.waveform.set_scale(self.settings.scale_uv)
        self.waveform.set_sample_rate(self.settings.sample_rate)
        self._sync_filter_controls()

    @staticmethod
    def _select_combo_data(combo: ComboBox, value) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return

    @staticmethod
    def _select_combo_text(combo: ComboBox, text: str) -> None:
        for index in range(combo.count()):
            if combo.itemText(index) == text:
                combo.setCurrentIndex(index)
                return

    def _connect_signals(self) -> None:
        self.refreshButton.clicked.connect(self._refresh_ports)
        self.connectButton.clicked.connect(self._toggle_connection)
        self.wifiCheck.toggled.connect(self._on_wifi_toggled)
        self.portCombo.currentIndexChanged.connect(self._on_port_changed)
        self.simButton.clicked.connect(self._toggle_simulator)
        self.streamButton.clicked.connect(self._toggle_stream)
        self.impedanceButton.clicked.connect(self._toggle_impedance)
        self.recordButton.clicked.connect(self._toggle_recording)
        self.markerButton.clicked.connect(self._add_marker)
        self.windowSpin.valueChanged.connect(self._apply_display_settings)
        self.scaleSpin.valueChanged.connect(self._apply_display_settings)
        self.waveform.scale_changed.connect(self._on_wave_scale_changed)
        self.filterSwitch.checkedChanged.connect(self._apply_filter_controls)
        self.lowSpin.valueChanged.connect(self._apply_filter_controls)
        self.highSpin.valueChanged.connect(self._apply_filter_controls)
        self.notchCombo.currentIndexChanged.connect(self._apply_filter_controls)
        self.rerefCombo.currentIndexChanged.connect(self._apply_rereference)
        self.referenceCombo.currentIndexChanged.connect(self._apply_rereference)

        self.controller.samples_ready.connect(self._on_samples)
        self.controller.connected_changed.connect(self._on_connected)
        self.controller.device_ready.connect(self._on_device_ready)
        self.controller.stats_updated.connect(self._on_stats)
        self.controller.impedance_updated.connect(self._on_impedance)
        self.controller.status_updated.connect(self._on_status)
        self.controller.recording_changed.connect(self._on_recording)
        self.controller.error.connect(self._on_error)

    # ------------------------------------------------------------------ 状态
    def _update_controls(self) -> None:
        connected = self.controller.connected
        wifi = self.wifiCheck.isChecked()
        self.connectButton.setText("断开" if connected else "连接")
        self.simButton.setText("退出模拟" if connected and self.controller.source == "simulator" else "模拟数据")
        self.streamButton.setEnabled(connected)
        self.impedanceButton.setEnabled(connected)
        self.recordButton.setEnabled(connected)
        self.markerButton.setEnabled(connected)
        # WiFi 模式下允许在连接中直接切换串口，切换时会自动断开并清理状态。
        self.portCombo.setEnabled(not connected or wifi)
        self.refreshButton.setEnabled(not connected or wifi)
        self.streamButton.setText("停止采集" if self._stream_running else "开始采集")
        self.impedanceButton.setText("停止阻抗" if self._impedance_running else "阻抗测试")
        self.recordButton.setText("停止记录" if self.controller.recording else "开始记录")
        self.statusPanel.set_mode(self._mode_text())

    def _mode_text(self) -> str:
        if not self.controller.connected:
            return "—"
        if self.controller.mode == "wifi":
            return f"WiFi 透传 {WIFI_SAMPLE_RATE} SPS / {self.controller.active_baudrate:,} bps"
        if self.controller.mode == "simulator":
            return "模拟数据"
        return f"USB CDC / {self.controller.active_baudrate:,} bps"

    def _update_mode_hint(self) -> None:
        if self.wifiCheck.isChecked():
            self.modeHint.setText(
                f"250 SPS / {self.settings.wifi_baudrate:,} bps（8N1）"
            )
        else:
            self.modeHint.setText("")

    def _ensure_wifi_sample_rate(self) -> None:
        """WiFi 模式固定 250 SPS，如与当前设置不一致则显式提示。"""

        if self.settings.sample_rate == WIFI_SAMPLE_RATE:
            return
        previous = self.settings.sample_rate
        self.settings.sample_rate = WIFI_SAMPLE_RATE
        self.waveform.set_sample_rate(WIFI_SAMPLE_RATE)
        self.controller.apply_config(WIFI_SAMPLE_RATE, self.settings.gain)
        notify(
            self,
            "采样率已限制",
            f"WiFi 透传仅为 {WIFI_SAMPLE_RATE} SPS，已从 {previous} Hz 切换为 {WIFI_SAMPLE_RATE} Hz",
            "warning",
        )

    def _reset_link(self) -> None:
        """关闭旧连接并清理接收队列、波形与状态显示。"""

        if self.controller.connected:
            self.controller.disconnect_device()
        self._stream_running = False
        self._impedance_running = False
        self.waveform.clear()
        self.impedancePanel.reset()
        self.statusPanel.set_connection(False, "")
        self.rateLabel.setText("-- Hz")

    # ------------------------------------------------------------------ 操作
    def _refresh_ports(self) -> None:
        self._load_ports()
        notify(self, "串口", f"发现 {self.portCombo.count()} 个串口", "info")

    def _on_wifi_toggled(self, checked: bool) -> None:
        self._reset_link()
        self.settings.wifi_mode = bool(checked)
        if checked:
            self._ensure_wifi_sample_rate()
            notify(
                self,
                "WiFi 透传模式",
                "请手动选择 WiFi 模块对应的串口后点击连接（默认 "
                f"{self.settings.wifi_baudrate:,} bps，{WIFI_SAMPLE_RATE} SPS）",
                "info",
            )
        self._update_mode_hint()
        self._update_controls()

    def _on_port_changed(self) -> None:
        if self._loading_ports:
            return
        self.settings.port = self.portCombo.currentData() or ""
        if self.controller.connected:
            self._reset_link()
            notify(self, "已断开", "切换串口后连接已关闭，接收缓存与状态已清理", "info")
        self._update_controls()

    def _toggle_connection(self) -> None:
        if self.controller.connected:
            self.controller.disconnect_device()
            return
        port = self.portCombo.currentData() or ""
        if not port:
            notify(self, "连接失败", "未发现可用串口", "error")
            return
        wifi = self.wifiCheck.isChecked()
        if wifi:
            self._ensure_wifi_sample_rate()
        baud = self.settings.wifi_baudrate if wifi else self.settings.baudrate
        self.settings.port = port
        self.controller.connect_serial(port, baud, auto_start=True, wifi=wifi)
        self._stream_running = True
        self.waveform.clear()
        self._update_controls()

    def _toggle_simulator(self) -> None:
        if self.controller.connected and self.controller.source == "simulator":
            self.controller.disconnect_device()
            return
        self.controller.connect_simulator(auto_start=True)
        self._stream_running = True
        self.waveform.clear()
        self._update_controls()

    def _toggle_stream(self) -> None:
        if self._stream_running:
            self.controller.stop_stream()
        else:
            self.controller.start_stream()
        self._stream_running = not self._stream_running
        self._update_controls()

    def _toggle_impedance(self) -> None:
        if self._impedance_running:
            self.controller.stop_impedance()
        else:
            self.controller.start_impedance()
        self._impedance_running = not self._impedance_running
        self._update_controls()

    def _toggle_recording(self) -> None:
        if self.controller.recording:
            path = self.controller.stop_recording()
            if path:
                notify(self, "记录已保存", path, "success")
        else:
            path = self.controller.start_recording()
            if path:
                notify(self, "开始记录", path, "success")
        self._update_controls()

    def _add_marker(self) -> None:
        self._marker_count += 1
        label = f"M{self._marker_count}"
        self.controller.add_marker(label)
        notify(self, "事件标记", f"已记录 {label}", "info")

    # ------------------------------------------------------------------ 设置
    def _apply_display_settings(self) -> None:
        self.settings.window_s = float(self.windowSpin.value())
        self.settings.scale_uv = float(self.scaleSpin.value())
        self.waveform.set_window(self.settings.window_s)
        self.waveform.set_scale(self.settings.scale_uv)

    def _on_wave_scale_changed(self, scale_uv: float) -> None:
        """滚轮缩放波形后同步「幅值」输入框。"""
        self.scaleSpin.blockSignals(True)
        self.scaleSpin.setValue(scale_uv)
        self.scaleSpin.blockSignals(False)
        self.settings.scale_uv = float(self.scaleSpin.value())

    def _sync_filter_controls(self) -> None:
        enabled = self.filterSwitch.isChecked()
        self.lowSpin.setEnabled(enabled)
        self.highSpin.setEnabled(enabled)
        self.notchCombo.setEnabled(enabled)

    def _apply_filter_controls(self) -> None:
        self._sync_filter_controls()
        notch = self.notchCombo.currentData() or 0
        self.controller.configure_filter(
            float(self.lowSpin.value()),
            float(self.highSpin.value()),
            int(notch),
            self.filterSwitch.isChecked(),
        )

    def _apply_rereference(self) -> None:
        mode = self.rerefCombo.currentData() or "none"
        channel = self.referenceCombo.currentText() if mode == "channel" else ""
        self.referenceCombo.setEnabled(mode == "channel")
        self.settings.reref_mode = mode
        self.settings.reref_channel = channel
        self.controller.set_rereference(mode, channel)

    def apply_settings(self, settings: Settings) -> None:
        """设置页保存后同步到本页。"""

        self.settings = settings
        if self.wifiCheck.isChecked() and settings.sample_rate != WIFI_SAMPLE_RATE:
            # 勾选 WiFi 模块时不允许其他采样率生效，明确提示而不是静默修改。
            settings.sample_rate = WIFI_SAMPLE_RATE
            notify(
                self,
                "采样率已限制",
                f"WiFi 透传仅支持 {WIFI_SAMPLE_RATE} SPS，设置页中的采样率不生效",
                "warning",
            )
        self.waveform.set_sample_rate(settings.sample_rate)
        self.controller.apply_config(settings.sample_rate, settings.gain)
        self._load_settings()
        self._apply_filter_controls()

    # ------------------------------------------------------------------ 回调
    def _on_samples(self, batch: SampleBatch) -> None:
        self.waveform.append(batch.values, batch.timestamps)

    def _on_connected(self, connected: bool, port: str) -> None:
        if connected:
            # 端口已打开但设备未必在线：先显示等待，收到有效帧后再标记连接成功。
            self.statusPanel.set_link_waiting(port)
        else:
            self._stream_running = False
            self._impedance_running = False
            self.impedancePanel.reset()
            self.statusPanel.set_connection(False, "")
        self._update_controls()

    def _on_device_ready(self, port: str) -> None:
        """收到设备首帧有效数据，确认链路可用。"""

        self.statusPanel.set_connection(True, port)
        notify(self, "已连接", port or "数据源就绪", "success")

    def _on_stats(self, stats) -> None:
        self.statusPanel.set_stats(stats)
        self.rateLabel.setText(f"{stats.sample_rate:.0f} Hz")

    def _on_impedance(self, values: np.ndarray) -> None:
        self.impedancePanel.update_values(values)

    def _on_status(self, status: DeviceStatus) -> None:
        self.statusPanel.set_status(status)

    def _on_recording(self, recording: bool, path: str) -> None:
        if recording:
            self.statusPanel.set_recording("记录中", "#c0392b")
        else:
            self.statusPanel.set_recording("已停止")
        self._update_controls()

    def _on_error(self, message: str) -> None:
        notify(self, "采集错误", message, "error")

    def _update_quality(self) -> None:
        if not self.controller.connected:
            return
        fs = max(self.settings.sample_rate, 1)
        values, _ = self.waveform.ring.snapshot(int(fs * 5))
        if values.shape[0] < fs:
            return
        results = evaluate_channels(values, fs, self.settings.normalized_channels())
        self.qualityTable.update_quality(results)

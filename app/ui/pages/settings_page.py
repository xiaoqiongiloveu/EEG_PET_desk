"""设置页面。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFileDialog, QGridLayout, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel,
    CardWidget,
    ComboBox,
    DoubleSpinBox,
    LineEdit,
    PrimaryPushButton,
    StrongBodyLabel,
    SwitchButton,
)

from ...config import SETTINGS_PATH, Settings, save_settings
from ...constants import (
    CHANNELS_RAW,
    SUPPORTED_BAUDRATES,
    SUPPORTED_GAINS,
    SUPPORTED_SAMPLE_RATES,
    WIFI_BAUDRATES,
    WIFI_SAMPLE_RATE,
)
from ..notify import notify

NOTCH_OPTIONS = (("关闭", 0), ("50 Hz", 50), ("60 Hz", 60))
REREF_OPTIONS = (("设备参考", "none"), ("平均参考 (CAR)", "car"), ("指定通道", "channel"))


class SettingsPage(QWidget):
    settings_saved = Signal(object)

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("settingsPage")
        self.settings = settings
        self._build_ui()
        self._load_from_settings()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        device = CardWidget(self)
        device_grid = QGridLayout(device)
        device_grid.setContentsMargins(14, 12, 14, 12)
        device_grid.addWidget(StrongBodyLabel("设备与采集", device), 0, 0, 1, 4)
        device_grid.addWidget(CaptionLabel("波特率", device), 1, 0)
        self.baudCombo = ComboBox(device)
        for baud in SUPPORTED_BAUDRATES:
            self.baudCombo.addItem(str(baud), userData=baud)
        device_grid.addWidget(self.baudCombo, 1, 1)
        device_grid.addWidget(CaptionLabel("采样率", device), 1, 2)
        self.rateCombo = ComboBox(device)
        for rate in SUPPORTED_SAMPLE_RATES:
            self.rateCombo.addItem(f"{rate} Hz", userData=rate)
        device_grid.addWidget(self.rateCombo, 1, 3)
        device_grid.addWidget(CaptionLabel("增益", device), 2, 0)
        self.gainCombo = ComboBox(device)
        for gain in SUPPORTED_GAINS:
            self.gainCombo.addItem(str(gain), userData=gain)
        device_grid.addWidget(self.gainCombo, 2, 1)
        device_grid.addWidget(
            CaptionLabel("修改采样率或增益会在下次连接时生效。", device), 2, 2, 1, 2
        )
        device_grid.addWidget(CaptionLabel("WiFi 波特率", device), 3, 0)
        self.wifiBaudCombo = ComboBox(device)
        for baud in WIFI_BAUDRATES:
            self.wifiBaudCombo.addItem(str(baud), userData=baud)
        device_grid.addWidget(self.wifiBaudCombo, 3, 1)
        device_grid.addWidget(
            CaptionLabel(
                f"勾选「使用 WiFi 模块」时按此波特率以 8N1 打开串口；"
                f"WiFi 透传固定 {WIFI_SAMPLE_RATE} SPS。",
                device,
            ),
            3,
            2,
            1,
            2,
        )
        layout.addWidget(device)

        display = CardWidget(self)
        display_grid = QGridLayout(display)
        display_grid.setContentsMargins(14, 12, 14, 12)
        display_grid.addWidget(StrongBodyLabel("显示与滤波", display), 0, 0, 1, 4)
        display_grid.addWidget(CaptionLabel("显示窗口", display), 1, 0)
        self.windowSpin = DoubleSpinBox(display)
        self.windowSpin.setRange(2.0, 60.0)
        self.windowSpin.setSuffix(" s")
        display_grid.addWidget(self.windowSpin, 1, 1)
        display_grid.addWidget(CaptionLabel("纵向幅值", display), 1, 2)
        self.scaleSpin = DoubleSpinBox(display)
        self.scaleSpin.setRange(10.0, 2000.0)
        self.scaleSpin.setSingleStep(10.0)
        self.scaleSpin.setSuffix(" µV")
        display_grid.addWidget(self.scaleSpin, 1, 3)
        display_grid.addWidget(CaptionLabel("高通", display), 2, 0)
        self.lowSpin = DoubleSpinBox(display)
        self.lowSpin.setRange(0.1, 20.0)
        self.lowSpin.setSingleStep(0.1)
        self.lowSpin.setSuffix(" Hz")
        display_grid.addWidget(self.lowSpin, 2, 1)
        display_grid.addWidget(CaptionLabel("低通", display), 2, 2)
        self.highSpin = DoubleSpinBox(display)
        self.highSpin.setRange(10.0, 120.0)
        self.highSpin.setSuffix(" Hz")
        display_grid.addWidget(self.highSpin, 2, 3)
        display_grid.addWidget(CaptionLabel("陷波", display), 3, 0)
        self.notchCombo = ComboBox(display)
        for text, value in NOTCH_OPTIONS:
            self.notchCombo.addItem(text, userData=value)
        display_grid.addWidget(self.notchCombo, 3, 1)
        self.filterSwitch = SwitchButton(display)
        display_grid.addWidget(CaptionLabel("启动时开启滤波", display), 3, 2)
        display_grid.addWidget(self.filterSwitch, 3, 3)
        layout.addWidget(display)

        reref = CardWidget(self)
        reref_grid = QGridLayout(reref)
        reref_grid.setContentsMargins(14, 12, 14, 12)
        reref_grid.addWidget(StrongBodyLabel("重参考", reref), 0, 0, 1, 4)
        reref_grid.addWidget(CaptionLabel("方式", reref), 1, 0)
        self.rerefCombo = ComboBox(reref)
        for text, value in REREF_OPTIONS:
            self.rerefCombo.addItem(text, userData=value)
        reref_grid.addWidget(self.rerefCombo, 1, 1)
        reref_grid.addWidget(CaptionLabel("参考通道", reref), 1, 2)
        self.referenceCombo = ComboBox(reref)
        self.referenceCombo.addItems(list(CHANNELS_RAW))
        reref_grid.addWidget(self.referenceCombo, 1, 3)
        layout.addWidget(reref)

        storage = CardWidget(self)
        storage_layout = QHBoxLayout(storage)
        storage_layout.setContentsMargins(14, 12, 14, 12)
        storage_layout.addWidget(CaptionLabel("存储目录", storage))
        self.storageEdit = LineEdit(storage)
        self.storageEdit.setPlaceholderText("留空表示使用程序目录下的 sessions")
        storage_layout.addWidget(self.storageEdit, 1)
        self.browseButton = PrimaryPushButton("选择…", storage)
        self.browseButton.clicked.connect(self._browse)
        storage_layout.addWidget(self.browseButton)
        layout.addWidget(storage)

        footer = QHBoxLayout()
        self.configLabel = CaptionLabel(f"配置文件：{SETTINGS_PATH}", self)
        footer.addWidget(self.configLabel, 1)
        self.saveButton = PrimaryPushButton("保存设置", self)
        self.saveButton.clicked.connect(self._save)
        footer.addWidget(self.saveButton)
        layout.addLayout(footer)
        layout.addStretch(1)

        self.rerefCombo.currentIndexChanged.connect(self._sync_reference)

    def _load_from_settings(self) -> None:
        select_data(self.baudCombo, self.settings.baudrate)
        select_data(self.wifiBaudCombo, self.settings.wifi_baudrate)
        select_data(self.rateCombo, self.settings.sample_rate)
        select_data(self.gainCombo, self.settings.gain)
        self.windowSpin.setValue(self.settings.window_s)
        self.scaleSpin.setValue(self.settings.scale_uv)
        self.lowSpin.setValue(self.settings.filter_low)
        self.highSpin.setValue(self.settings.filter_high)
        select_data(self.notchCombo, self.settings.notch)
        self.filterSwitch.setChecked(self.settings.filter_enabled)
        select_data(self.rerefCombo, self.settings.reref_mode)
        for index in range(self.referenceCombo.count()):
            if self.referenceCombo.itemText(index) == self.settings.reref_channel:
                self.referenceCombo.setCurrentIndex(index)
                break
        self.storageEdit.setText(self.settings.storage_dir)
        self._sync_reference()

    def _sync_reference(self) -> None:
        self.referenceCombo.setEnabled(self.rerefCombo.currentData() == "channel")

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择存储目录", self.storageEdit.text() or str(Path.home()))
        if path:
            self.storageEdit.setText(path)

    def _save(self) -> None:
        settings = self.settings
        settings.baudrate = int(self.baudCombo.currentData())
        settings.wifi_baudrate = int(self.wifiBaudCombo.currentData())
        settings.sample_rate = int(self.rateCombo.currentData())
        settings.gain = int(self.gainCombo.currentData())
        settings.window_s = float(self.windowSpin.value())
        settings.scale_uv = float(self.scaleSpin.value())
        settings.filter_low = float(self.lowSpin.value())
        settings.filter_high = float(self.highSpin.value())
        settings.notch = int(self.notchCombo.currentData() or 0)
        settings.filter_enabled = self.filterSwitch.isChecked()
        settings.reref_mode = self.rerefCombo.currentData() or "none"
        settings.reref_channel = (
            self.referenceCombo.currentText() if settings.reref_mode == "channel" else ""
        )
        settings.storage_dir = self.storageEdit.text().strip()
        settings.replay_filter_low = settings.filter_low
        settings.replay_filter_high = settings.filter_high
        settings.replay_notch = settings.notch
        try:
            save_settings(settings)
        except OSError as exc:
            notify(self, "保存失败", str(exc), "error")
            return
        self.settings_saved.emit(settings)
        notify(self, "设置已保存", str(SETTINGS_PATH), "success")


def select_data(combo: ComboBox, value) -> None:
    for index in range(combo.count()):
        if combo.itemData(index) == value:
            combo.setCurrentIndex(index)
            return

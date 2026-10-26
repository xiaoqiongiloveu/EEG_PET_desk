"""回放与分析页面。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QListWidget,
    QListWidgetItem,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    ComboBox,
    DoubleSpinBox,
    FluentIcon,
    PrimaryPushButton,
    PushButton,
    StrongBodyLabel,
    SwitchButton,
)

from ...analysis.quality import artifact_ratio, evaluate_channels
from ...analysis.spectrum import analyze_spectrum
from ...analysis.stats import channel_statistics
from ...config import Settings
from ...core.filters import apply_rereference, offline_filter
from ...storage.session import SessionInfo, SessionReader, list_sessions
from ..notify import notify
from ..widgets.panels import QualityTable
from ..widgets.waveform import StaticWaveformWidget

NOTCH_OPTIONS = (("关闭", 0), ("50 Hz", 50), ("60 Hz", 60))
REREF_OPTIONS = (("原始参考", "none"), ("平均参考 (CAR)", "car"), ("指定通道", "channel"))
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)


class ReplayPage(QWidget):
    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("replayPage")
        self.settings = settings
        self.reader: SessionReader | None = None
        self._raw = np.empty((0, 8), dtype=np.float32)
        self._timestamps = np.empty((0,), dtype=np.float64)
        self._signal = np.empty((0, 8), dtype=np.float32)
        self._cursor_s = 0.0
        self._playing = False
        self._updating_slider = False

        self._build_ui()
        self._refresh_sessions()

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_tick)
        self._play_timer.start(50)

    # ------------------------------------------------------------------ 界面
    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        left = CardWidget(self)
        left.setFixedWidth(300)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        title_row = QHBoxLayout()
        title_row.addWidget(StrongBodyLabel("历史会话", left))
        title_row.addStretch(1)
        self.refreshButton = PushButton(FluentIcon.SYNC, "刷新", left)
        title_row.addWidget(self.refreshButton)
        left_layout.addLayout(title_row)
        self.sessionList = QListWidget(left)
        left_layout.addWidget(self.sessionList, 1)
        self.sessionInfo = CaptionLabel("选择一条会话记录", left)
        self.sessionInfo.setWordWrap(True)
        left_layout.addWidget(self.sessionInfo)
        layout.addWidget(left)

        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        self.playButton = PrimaryPushButton(FluentIcon.PLAY, "播放", self)
        toolbar.addWidget(self.playButton)
        self.positionSlider = QSlider(Qt.Horizontal, self)
        self.positionSlider.setRange(0, 1000)
        toolbar.addWidget(self.positionSlider, 1)
        self.timeLabel = CaptionLabel("0.0 / 0.0 s", self)
        toolbar.addWidget(self.timeLabel)

        toolbar.addWidget(CaptionLabel("倍速", self))
        self.speedCombo = ComboBox(self)
        for speed in SPEEDS:
            self.speedCombo.addItem(f"{speed:g}×", userData=speed)
        self.speedCombo.setCurrentIndex(2)
        toolbar.addWidget(self.speedCombo)

        toolbar.addWidget(CaptionLabel("窗口", self))
        self.windowSpin = DoubleSpinBox(self)
        self.windowSpin.setRange(1.0, 60.0)
        self.windowSpin.setSuffix(" s")
        self.windowSpin.setValue(settings_window(self.settings))
        toolbar.addWidget(self.windowSpin)

        self.filterSwitch = SwitchButton(self)
        self.filterSwitch.setChecked(True)
        toolbar.addWidget(CaptionLabel("零相位滤波", self))
        toolbar.addWidget(self.filterSwitch)
        toolbar.addWidget(CaptionLabel("高通", self))
        self.lowSpin = DoubleSpinBox(self)
        self.lowSpin.setRange(0.1, 20.0)
        self.lowSpin.setSingleStep(0.1)
        self.lowSpin.setSuffix(" Hz")
        self.lowSpin.setValue(self.settings.replay_filter_low)
        toolbar.addWidget(self.lowSpin)
        toolbar.addWidget(CaptionLabel("低通", self))
        self.highSpin = DoubleSpinBox(self)
        self.highSpin.setRange(10.0, 120.0)
        self.highSpin.setSuffix(" Hz")
        self.highSpin.setValue(self.settings.replay_filter_high)
        toolbar.addWidget(self.highSpin)
        toolbar.addWidget(CaptionLabel("陷波", self))
        self.notchCombo = ComboBox(self)
        for text, value in NOTCH_OPTIONS:
            self.notchCombo.addItem(text, userData=value)
        select_combo_data(self.notchCombo, self.settings.replay_notch)
        toolbar.addWidget(self.notchCombo)
        right_layout.addLayout(toolbar)

        self.rerefBar = QHBoxLayout()
        self.rerefBar.setSpacing(6)
        self.rerefBar.addWidget(CaptionLabel("重参考", self))
        self.rerefCombo = ComboBox(self)
        for text, value in REREF_OPTIONS:
            self.rerefCombo.addItem(text, userData=value)
        self.rerefBar.addWidget(self.rerefCombo)
        self.referenceCombo = ComboBox(self)
        self.referenceCombo.setEnabled(False)
        self.rerefBar.addWidget(self.referenceCombo)
        self.rerefBar.addStretch(1)
        self.analyzeWindowButton = PushButton("分析当前窗口", self)
        self.analyzeAllButton = PushButton("分析全部", self)
        self.rerefBar.addWidget(self.analyzeWindowButton)
        self.rerefBar.addWidget(self.analyzeAllButton)
        self.cursorLabel = CaptionLabel("点击波形可读取游标数值", self)
        self.rerefBar.addWidget(self.cursorLabel)
        right_layout.addLayout(self.rerefBar)

        self.waveform = StaticWaveformWidget(250, list(self.settings.normalized_channels()), parent=self)
        right_layout.addWidget(self.waveform, 3)

        self.analysisTabs = QTabWidget(self)
        self.spectrumWidget = pg.PlotWidget(background="#ffffff")
        self.spectrumWidget.setMenuEnabled(False)
        self.spectrumWidget.showGrid(x=True, y=True, alpha=0.15)
        self.spectrumWidget.setLabel("bottom", "频率", units="Hz")
        self.spectrumWidget.setLabel("left", "PSD", units="µV²/Hz")
        self.spectrumWidget.setLogMode(y=True)
        self.analysisTabs.addTab(self.spectrumWidget, "功率谱")

        self.statsTable = QTableWidget(0, 5, self)
        self.statsTable.setHorizontalHeaderLabels(["通道", "均值(µV)", "标准差(µV)", "RMS(µV)", "峰峰值(µV)"])
        self.statsTable.verticalHeader().setVisible(False)
        self.statsTable.setEditTriggers(QTableWidget.NoEditTriggers)
        self.statsTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.analysisTabs.addTab(self.statsTable, "统计")

        self.qualityTable = QualityTable(self)
        self.analysisTabs.addTab(self.qualityTable, "质量")
        right_layout.addWidget(self.analysisTabs, 2)

        layout.addWidget(right, 1)

        self.refreshButton.clicked.connect(self._refresh_sessions)
        self.sessionList.itemSelectionChanged.connect(self._on_session_selected)
        self.playButton.clicked.connect(self._toggle_play)
        self.positionSlider.sliderPressed.connect(self._on_slider_pressed)
        self.positionSlider.valueChanged.connect(self._on_slider_changed)
        self.speedCombo.currentIndexChanged.connect(lambda _: None)
        self.windowSpin.valueChanged.connect(self._on_window_changed)
        self.filterSwitch.checkedChanged.connect(self._on_filter_changed)
        self.lowSpin.valueChanged.connect(self._on_filter_changed)
        self.highSpin.valueChanged.connect(self._on_filter_changed)
        self.notchCombo.currentIndexChanged.connect(self._on_filter_changed)
        self.rerefCombo.currentIndexChanged.connect(self._on_reref_changed)
        self.referenceCombo.currentIndexChanged.connect(self._on_reref_changed)
        self.analyzeWindowButton.clicked.connect(lambda: self._analyze(window=True))
        self.analyzeAllButton.clicked.connect(lambda: self._analyze(window=False))
        self.waveform.plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

    # ------------------------------------------------------------------ 会话
    def _refresh_sessions(self) -> None:
        self.sessionList.clear()
        root = self.settings.resolved_storage_dir()
        for info in list_sessions(root):
            item = QListWidgetItem(info.summary())
            item.setData(Qt.UserRole, str(info.path))
            self.sessionList.addItem(item)
        if self.sessionList.count() == 0:
            self.sessionInfo.setText(f"{root} 下没有会话记录")

    def _on_session_selected(self) -> None:
        items = self.sessionList.selectedItems()
        if not items:
            return
        self._load_session(Path(items[0].data(Qt.UserRole)))

    def _load_session(self, path: Path) -> None:
        try:
            reader = SessionReader(path)
            values, timestamps = reader.read()
        except OSError as exc:
            notify(self, "加载失败", str(exc), "error")
            return
        self.reader = reader
        self._raw = values
        self._timestamps = timestamps
        info = reader.info
        self.sessionInfo.setText(
            f"{info.name}\n{info.created_at.replace('T', ' ')}\n"
            f"{reader.duration_s:.1f}s  {reader.sample_rate}Hz  {reader.channels}ch\n"
            f"增益 {info.gain}，来源 {info.metadata.get('source', '未知')}"
        )
        self.waveform.set_sample_rate(reader.sample_rate)
        self.waveform.channel_names = [str(name) for name in info.channel_names]
        self.waveform.order = list(range(len(info.channel_names)))[: len(info.channel_names)]
        self.waveform.names = self.waveform.channel_names
        self.waveform.set_scale(self.settings.scale_uv)
        self.referenceCombo.clear()
        self.referenceCombo.addItems(self.waveform.channel_names)
        self._rebuild_signal()
        self._cursor_s = min(self.windowSpin.value(), reader.duration_s)
        self._refresh_view()
        self._clear_analysis()
        notify(self, "已加载", f"{info.name}，{reader.duration_s:.1f} 秒", "success")

    # ------------------------------------------------------------------ 信号
    def _rebuild_signal(self) -> None:
        if self.reader is None or self._raw.size == 0:
            self._signal = np.empty((0, 8), dtype=np.float32)
            return
        mode = self.rerefCombo.currentData() or "none"
        channel = self.referenceCombo.currentText() if mode == "channel" else ""
        self.referenceCombo.setEnabled(mode == "channel")
        values = apply_rereference(
            self._raw,
            self.waveform.channel_names,
            mode,
            channel,
        )
        if self.filterSwitch.isChecked():
            values = offline_filter(
                values,
                self.reader.sample_rate,
                float(self.lowSpin.value()),
                float(self.highSpin.value()),
                int(self.notchCombo.currentData() or 0),
            )
        self._signal = values
        self.waveform.set_data(self._signal, self._timestamps)

    def _refresh_view(self) -> None:
        if self.reader is None:
            return
        duration = max(self.reader.duration_s, 0.001)
        self._cursor_s = float(np.clip(self._cursor_s, 0.0, duration))
        self.waveform.set_cursor(self._cursor_s)
        self.timeLabel.setText(f"{self._cursor_s:7.2f} / {duration:7.2f} s")
        self._updating_slider = True
        self.positionSlider.setValue(int(self._cursor_s / duration * 1000))
        self._updating_slider = False

    def _on_window_changed(self) -> None:
        self.waveform.set_window(float(self.windowSpin.value()))
        self._refresh_view()

    def _on_filter_changed(self) -> None:
        if self.reader is not None:
            self._rebuild_signal()
            self._refresh_view()

    def _on_reref_changed(self) -> None:
        if self.reader is not None:
            self._rebuild_signal()
            self._refresh_view()

    # ------------------------------------------------------------------ 播放
    def _toggle_play(self) -> None:
        if self.reader is None:
            notify(self, "播放", "请先选择一条会话记录", "warning")
            return
        if self._playing:
            self._playing = False
            self.playButton.setText("播放")
            return
        if self._cursor_s >= self.reader.duration_s - 0.01:
            self._cursor_s = 0.0
        self._playing = True
        self.playButton.setText("暂停")

    def _on_tick(self) -> None:
        if not self._playing or self.reader is None:
            return
        speed = float(self.speedCombo.currentData() or 1.0)
        self._cursor_s += 0.05 * speed
        if self._cursor_s >= self.reader.duration_s:
            self._cursor_s = self.reader.duration_s
            self._playing = False
            self.playButton.setText("播放")
        self._refresh_view()

    def _on_slider_pressed(self) -> None:
        self._playing = False
        self.playButton.setText("播放")

    def _on_slider_changed(self, value: int) -> None:
        if self._updating_slider or self.reader is None:
            return
        self._cursor_s = self.reader.duration_s * value / 1000.0
        self._refresh_view()

    # ------------------------------------------------------------------ 分析
    def _window_data(self) -> tuple[np.ndarray, np.ndarray]:
        if self.reader is None:
            return np.empty((0, 8), dtype=np.float32), np.empty((0,), dtype=np.float64)
        window = float(self.windowSpin.value())
        end = self._cursor_s
        start = max(end - window, float(self._timestamps[0]) if len(self._timestamps) else 0.0)
        left = int(np.searchsorted(self._timestamps, start, side="left"))
        right = int(np.searchsorted(self._timestamps, end, side="right"))
        return self._signal[left:right], self._timestamps[left:right]

    def _analyze(self, window: bool = True) -> None:
        if self.reader is None:
            notify(self, "分析", "请先加载会话", "warning")
            return
        if window:
            values, timestamps = self._window_data()
        else:
            values, timestamps = self._signal, self._timestamps
        if len(values) < 16:
            notify(self, "分析", "当前数据太短，无法分析", "warning")
            return
        fs = self.reader.sample_rate
        names = self.waveform.channel_names

        spectrum = analyze_spectrum(values, fs, nperseg_s=min(2.0, len(values) / fs))
        self.spectrumWidget.clear()
        limit = min(45.0, fs / 2.0)
        mask = spectrum.freqs <= limit
        for index in range(values.shape[1]):
            color = pg.intColor(index, hues=8)
            self.spectrumWidget.plot(
                spectrum.freqs[mask],
                spectrum.psd[index, mask],
                pen=pg.mkPen(color, width=1),
            )
        self.analysisTabs.setCurrentIndex(0)
        self.spectrumWidget.setXRange(0, limit, padding=0)

        stats = channel_statistics(values)
        self.statsTable.setRowCount(len(stats))
        for row, item in enumerate(stats):
            values_row = (
                names[row] if row < len(names) else f"CH{row + 1}",
                f"{item.mean_uv:.2f}",
                f"{item.std_uv:.2f}",
                f"{item.rms_uv:.2f}",
                f"{item.p2p_uv:.1f}",
            )
            for column, text in enumerate(values_row):
                cell = QTableWidgetItem(text)
                cell.setTextAlignment(Qt.AlignCenter)
                self.statsTable.setItem(row, column, cell)

        quality = evaluate_channels(values, fs, names)
        self.qualityTable.update_quality(quality)
        ratio = artifact_ratio(values)
        self.cursorLabel.setText(f"伪迹比例 {ratio * 100:.2f}%（σ>6）")

    def _clear_analysis(self) -> None:
        self.spectrumWidget.clear()
        self.statsTable.setRowCount(0)
        self.qualityTable.table.setRowCount(0)
        self.cursorLabel.setText("点击波形可读取游标数值")

    def _on_plot_clicked(self, event) -> None:
        if self.reader is None or self._signal.size == 0:
            return
        view = self.waveform.plot.getPlotItem().vb
        point = view.mapSceneToView(event.scenePos())
        t = float(point.x())
        index = int(np.clip(np.searchsorted(self._timestamps, t), 0, len(self._timestamps) - 1))
        values = self._signal[index]
        pairs = "  ".join(
            f"{name}={value:.0f}"
            for name, value in zip(self.waveform.channel_names, values)
        )
        self.cursorLabel.setText(f"t={self._timestamps[index]:.2f}s  {pairs} µV")


def settings_window(settings: Settings) -> float:
    return float(getattr(settings, "replay_window_s", 10.0))


def select_combo_data(combo: ComboBox, value) -> None:
    for index in range(combo.count()):
        if combo.itemData(index) == value:
            combo.setCurrentIndex(index)
            return

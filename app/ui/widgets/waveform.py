"""实时与回放波形控件（pyqtgraph）。"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ...constants import RING_SECONDS
from ...core.ring_buffer import MultiChannelRingBuffer

COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
)

# 滚轮缩放范围与「幅值」输入框保持一致。
MIN_SCALE_UV = 10.0
MAX_SCALE_UV = 2000.0


class _WaveformBase(QWidget):
    scale_changed = Signal(float)

    def __init__(self, sample_rate: int, channel_names: list[str], order: list[int] | None, parent=None) -> None:
        super().__init__(parent)
        self.sample_rate = max(int(sample_rate), 1)
        self.channel_names = list(channel_names)
        self.order = list(order) if order else list(range(len(self.channel_names)))
        self.names = [
            self.channel_names[i] if i < len(self.channel_names) else f"CH{i + 1}"
            for i in self.order
        ]
        self.scale_uv = 100.0
        self.window_s = 10.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget(background="#ffffff")
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.showGrid(x=True, y=False, alpha=0.15)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.getAxis("left").setWidth(58)
        self.plot.getAxis("left").setPen(pg.mkPen("#c9c9c9"))
        self.plot.getAxis("bottom").setPen(pg.mkPen("#c9c9c9"))
        self.curves: list[pg.PlotDataItem] = []
        for index, color in enumerate(COLORS[: len(self.names)]):
            self.curves.append(self.plot.plot([], [], pen=pg.mkPen(color, width=1)))
        layout.addWidget(self.plot)

        # 每个通道名字上方显示窗口内最大值、下方显示最小值（无单位、取整）。
        self._max_labels: list[pg.TextItem] = []
        self._min_labels: list[pg.TextItem] = []
        font = QFont()
        font.setPointSize(7)
        for _ in self.names:
            for bucket in (self._max_labels, self._min_labels):
                item = pg.TextItem("", color="#9a9a9a", anchor=(0, 0.5))
                item.setFont(font)
                self.plot.addItem(item)
                bucket.append(item)

        self.plot.viewport().installEventFilter(self)
        self._apply_scale()

    def _offsets(self) -> np.ndarray:
        count = len(self.names)
        return np.array([(count - 1 - i) * self.scale_uv for i in range(count)], dtype=np.float64)

    def _apply_scale(self) -> None:
        offsets = self._offsets()
        ticks = [(float(value), name) for value, name in zip(offsets, self.names)]
        axis = self.plot.getAxis("left")
        axis.setTicks([ticks])
        span = max(self.scale_uv, 1.0)
        self.plot.setYRange(
            offsets.min() - 0.55 * span,
            offsets.max() + 0.55 * span,
            padding=0,
        )

    def set_scale(self, scale_uv: float) -> None:
        scale_uv = float(np.clip(float(scale_uv), MIN_SCALE_UV, MAX_SCALE_UV))
        if abs(scale_uv - self.scale_uv) < 1e-6:
            return
        self.scale_uv = scale_uv
        self._apply_scale()
        self.refresh()

    def eventFilter(self, obj, event) -> bool:
        """鼠标位于波形区域时，滚轮上下缩放各通道的 y 轴比例。"""
        if obj is self.plot.viewport() and event.type() == QEvent.Wheel:
            delta = event.angleDelta().y() or event.pixelDelta().y()
            if delta:
                self._zoom_by_steps(delta / 120.0)
            return True
        return super().eventFilter(obj, event)

    def _zoom_by_steps(self, steps: float) -> None:
        # 向上滚动放大波形（幅值比例变小），向下滚动缩小。
        self.set_scale(self.scale_uv / (1.15 ** steps))
        self.scale_changed.emit(self.scale_uv)

    def _update_value_labels(self, window_values: np.ndarray | None) -> None:
        if window_values is None or window_values.shape[0] == 0:
            for item in self._max_labels + self._min_labels:
                item.setText("")
            return
        offsets = self._offsets()
        left, right = self.plot.viewRange()[0]
        x = left + 0.006 * max(right - left, 1e-6)
        # 用像素换算上下间距，保证缩放时标签始终紧贴通道名。
        dy = 9.0 * abs(self.plot.getPlotItem().vb.viewPixelSize()[1])
        for index, channel in enumerate(self.order):
            if channel >= window_values.shape[1]:
                continue
            data = window_values[:, channel]
            self._max_labels[index].setText(str(int(np.round(float(np.max(data))))))
            self._min_labels[index].setText(str(int(np.round(float(np.min(data))))))
            self._max_labels[index].setPos(x, offsets[index] + dy)
            self._min_labels[index].setPos(x, offsets[index] - dy)

    def set_window(self, seconds: float) -> None:
        self.window_s = max(float(seconds), 0.5)
        self.refresh()

    def set_sample_rate(self, sample_rate: int) -> None:
        self.sample_rate = max(int(sample_rate), 1)

    def refresh(self) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        for curve in self.curves:
            curve.clear()
        self._update_value_labels(None)


class LiveWaveformWidget(_WaveformBase):
    """带环形缓冲的滚动显示。"""

    def __init__(self, sample_rate: int, channel_names: list[str], order: list[int] | None = None, parent=None) -> None:
        super().__init__(sample_rate, channel_names, order, parent)
        self.ring = MultiChannelRingBuffer(int(sample_rate * RING_SECONDS), len(self.channel_names))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(33)

    def append(self, values: np.ndarray, timestamps: np.ndarray) -> None:
        self.ring.append(values, timestamps)

    def set_sample_rate(self, sample_rate: int) -> None:
        sample_rate = max(int(sample_rate), 1)
        if sample_rate == self.sample_rate:
            return
        super().set_sample_rate(sample_rate)
        self.ring = MultiChannelRingBuffer(int(sample_rate * RING_SECONDS), len(self.channel_names))

    def clear(self) -> None:
        super().clear()
        self.ring.clear()

    def refresh(self) -> None:
        need = int(self.window_s * self.sample_rate) + 4
        values, timestamps = self.ring.snapshot(need)
        if values.shape[0] < 2:
            _WaveformBase.clear(self)
            return
        x = timestamps - timestamps[-1]
        offsets = self._offsets()
        for index, curve in enumerate(self.curves):
            curve.setData(x, values[:, self.order[index]] + offsets[index])
        self.plot.setXRange(-self.window_s, 0, padding=0)
        visible = max(int(self.window_s * self.sample_rate), 2)
        self._update_value_labels(values[-visible:])


class StaticWaveformWidget(_WaveformBase):
    """回放模式：显示指定时间窗内的数据。"""

    def __init__(self, sample_rate: int, channel_names: list[str], order: list[int] | None = None, parent=None) -> None:
        super().__init__(sample_rate, channel_names, order, parent)
        self._values = np.empty((0, len(self.channel_names)), dtype=np.float32)
        self._timestamps = np.empty((0,), dtype=np.float64)
        self._cursor_s = 0.0

    def set_data(self, values: np.ndarray, timestamps: np.ndarray) -> None:
        self._values = np.asarray(values, dtype=np.float32)
        self._timestamps = np.asarray(timestamps, dtype=np.float64)

    def set_cursor(self, cursor_s: float) -> None:
        self._cursor_s = float(cursor_s)
        self.refresh()

    def refresh(self) -> None:
        if self._values.shape[0] < 2:
            self.clear()
            return
        start = max(self._cursor_s - self.window_s, float(self._timestamps[0]))
        end = max(self._cursor_s, start + 0.01)
        left = int(np.searchsorted(self._timestamps, start, side="left"))
        right = int(np.searchsorted(self._timestamps, end, side="right"))
        if right - left < 2:
            self.clear()
            return
        window_values = self._values[left:right]
        x = self._timestamps[left:right]
        offsets = self._offsets()
        for index, curve in enumerate(self.curves):
            curve.setData(x, window_values[:, self.order[index]] + offsets[index])
        self.plot.setXRange(start, end, padding=0)
        self.plot.setYRange(
            offsets.min() - 0.55 * self.scale_uv,
            offsets.max() + 0.55 * self.scale_uv,
            padding=0,
        )
        self._update_value_labels(window_values)

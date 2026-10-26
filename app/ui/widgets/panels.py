"""阻抗、设备状态与信号质量面板。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import BodyLabel, CardWidget, CaptionLabel, StrongBodyLabel

from ...analysis.quality import ChannelQuality
from ...constants import DEVICE_STATE_NAMES
from ...core.models import AcquisitionStats
from ...protocol import DeviceStatus

GOOD_COLOR = "#2ea44f"
WARN_COLOR = "#e6a23c"
BAD_COLOR = "#e74c3c"


def quality_color(value: float) -> str:
    if value < 5.0:
        return GOOD_COLOR
    if value < 20.0:
        return WARN_COLOR
    return BAD_COLOR


class _ImpedanceCell(QWidget):
    def __init__(self, name: str, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(2)
        self.name_label = StrongBodyLabel(name, self)
        self.value_label = CaptionLabel("-- kΩ", self)
        self.bar = QProgressBar(self)
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        layout.addWidget(self.name_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.bar)

    def set_value(self, value: float) -> None:
        color = quality_color(value)
        self.value_label.setText(f"{value:.1f} kΩ")
        self.value_label.setStyleSheet(f"color: {color};")
        self.bar.setValue(int(min(max(value, 0.0), 100.0)))
        self.bar.setStyleSheet(
            "QProgressBar { background: #ececec; border-radius: 3px; }"
            f"QProgressBar::chunk {{ background: {color}; border-radius: 3px; }}"
        )


class ImpedancePanel(CardWidget):
    def __init__(self, channel_names: list[str], order: list[int], parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        title = StrongBodyLabel("电极阻抗", self)
        layout.addWidget(title)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        self.cells: dict[int, _ImpedanceCell] = {}
        for position, raw_index in enumerate(order):
            name = channel_names[raw_index] if raw_index < len(channel_names) else f"CH{raw_index + 1}"
            cell = _ImpedanceCell(name, self)
            grid.addWidget(cell, position // 4, position % 4)
            self.cells[raw_index] = cell
        layout.addLayout(grid)

    def update_values(self, values: np.ndarray) -> None:
        for index, cell in self.cells.items():
            if index < len(values):
                cell.set_value(float(values[index]))

    def reset(self) -> None:
        for cell in self.cells.values():
            cell.value_label.setText("-- kΩ")
            cell.bar.setValue(0)


class StatusPanel(CardWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        layout.addWidget(StrongBodyLabel("设备状态", self))
        self.labels: dict[str, BodyLabel] = {}
        for key in ("连接", "模式", "设备", "电量", "SD 卡", "数据率", "丢包", "CRC 错误", "记录"):
            row = QHBoxLayout()
            caption = CaptionLabel(f"{key}", self)
            caption.setFixedWidth(64)
            value = BodyLabel("--", self)
            row.addWidget(caption)
            row.addWidget(value, 1)
            layout.addLayout(row)
            self.labels[key] = value

    def _set(self, key: str, text: str, color: str = "") -> None:
        label = self.labels[key]
        label.setText(text)
        label.setStyleSheet(f"color: {color};" if color else "")

    def set_connection(self, connected: bool, port: str) -> None:
        if connected:
            self._set("连接", f"已连接 {port}" if port else "已连接", GOOD_COLOR)
        else:
            self._set("连接", "未连接", WARN_COLOR)

    def set_link_waiting(self, port: str) -> None:
        """串口已打开但尚未收到设备有效响应。"""

        detail = f"{port} 已打开，等待设备响应…" if port else "等待设备响应…"
        self._set("连接", detail, WARN_COLOR)

    def set_mode(self, text: str, color: str = "") -> None:
        self._set("模式", text, color)

    def set_status(self, status: DeviceStatus) -> None:
        state = DEVICE_STATE_NAMES.get(status.state_code, f"状态 {status.state_code}")
        self._set("设备", f"{state}（固件 {status.firmware or '未知'}）")
        self._set("电量", f"{status.battery_v:.2f} V")
        self._set("SD 卡", f"{'正常' if status.sd_ok else '异常'}，使用率 {status.sd_usage_percent}%")

    def set_stats(self, stats: AcquisitionStats) -> None:
        self._set("数据率", f"{stats.sample_rate:.1f} Hz / {stats.frame_rate:.1f} 帧/s")
        drop_color = GOOD_COLOR if stats.sequence_gaps == 0 else WARN_COLOR
        self._set(
            "丢包",
            f"缺口 {stats.sequence_gaps}，丢样 {stats.dropped_samples}",
            drop_color,
        )
        crc_color = GOOD_COLOR if stats.crc_errors == 0 else BAD_COLOR
        self._set(
            "CRC 错误",
            f"{stats.crc_errors}（畸形帧 {stats.malformed_frames}）",
            crc_color,
        )

    def set_recording(self, text: str, color: str = "") -> None:
        self._set("记录", text, color)


class QualityTable(CardWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.addWidget(StrongBodyLabel("信号质量", self))
        self.table = QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(["通道", "评分", "RMS(µV)", "问题"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        layout.addWidget(self.table)

    def update_quality(self, results: list[ChannelQuality]) -> None:
        self.table.setRowCount(len(results))
        for row, item in enumerate(results):
            name = QTableWidgetItem(item.name)
            name.setTextAlignment(Qt.AlignCenter)
            score = QTableWidgetItem(str(item.score))
            score.setTextAlignment(Qt.AlignCenter)
            score.setForeground(QColor("#c0392b") if item.score < 60 else QColor("#1b1b1b"))
            rms = QTableWidgetItem(f"{item.rms_uv:.1f}")
            rms.setTextAlignment(Qt.AlignCenter)
            flags = QTableWidgetItem("、".join(item.flags) if item.flags else "正常")
            flags.setForeground(QColor("#c0392b") if item.flags else QColor("#2e7d32"))
            self.table.setItem(row, 0, name)
            self.table.setItem(row, 1, score)
            self.table.setItem(row, 2, rms)
            self.table.setItem(row, 3, flags)

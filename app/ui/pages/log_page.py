"""命令与日志页面。"""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CardWidget,
    CaptionLabel,
    PrimaryPushButton,
    PushButton,
    StrongBodyLabel,
)

from ...constants import (
    CMD_CLEAR_DATA,
    CMD_DISABLE_DEVICE,
    CMD_ENABLE_DEVICE,
    CMD_GET_STATE,
    CMD_PAUSE,
    CMD_RESUME,
    CMD_START_STREAM,
    CMD_STOP_STREAM,
    COMMAND_NAMES,
)
from ...controller import AcquisitionController
from ...protocol import CommandResponse
from ..notify import notify


class LogPage(QWidget):
    def __init__(self, controller: AcquisitionController, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("logPage")
        self.controller = controller
        self._build_ui()
        self.controller.log.connect(self.append_log)
        self.controller.command_response.connect(self._on_response)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        commands = CardWidget(self)
        command_layout = QVBoxLayout(commands)
        command_layout.setContentsMargins(12, 10, 12, 10)
        command_layout.addWidget(StrongBodyLabel("设备命令", commands))
        row = QHBoxLayout()
        row.setSpacing(6)
        self._add_command_button(row, "查询状态", CMD_GET_STATE, commands)
        self._add_command_button(row, "使能设备", CMD_ENABLE_DEVICE, commands)
        self._add_command_button(row, "停用设备", CMD_DISABLE_DEVICE, commands)
        self._add_command_button(row, "开始采集", CMD_START_STREAM, commands)
        self._add_command_button(row, "停止采集", CMD_STOP_STREAM, commands)
        self._add_command_button(row, "暂停", CMD_PAUSE, commands)
        self._add_command_button(row, "恢复", CMD_RESUME, commands)
        self.clearDataButton = PushButton("清空 SD 数据", commands)
        self.clearDataButton.clicked.connect(self._confirm_clear_data)
        row.addWidget(self.clearDataButton)
        row.addStretch(1)
        command_layout.addLayout(row)
        hint = CaptionLabel("命令会通过串口发送，响应显示在下方表格与日志中。", commands)
        command_layout.addWidget(hint)
        layout.addWidget(commands)

        responses = CardWidget(self)
        response_layout = QVBoxLayout(responses)
        response_layout.setContentsMargins(12, 10, 12, 10)
        header = QHBoxLayout()
        header.addWidget(StrongBodyLabel("命令响应", responses))
        header.addStretch(1)
        self.clearButton = PushButton("清空日志", responses)
        self.clearButton.clicked.connect(self._clear_all)
        header.addWidget(self.clearButton)
        response_layout.addLayout(header)
        self.responseTable = QTableWidget(0, 4, responses)
        self.responseTable.setHorizontalHeaderLabels(["时间", "命令", "结果", "说明"])
        self.responseTable.verticalHeader().setVisible(False)
        self.responseTable.setEditTriggers(QTableWidget.NoEditTriggers)
        self.responseTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.responseTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.responseTable.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.responseTable.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        response_layout.addWidget(self.responseTable)
        layout.addWidget(responses, 1)

        log_card = CardWidget(self)
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(12, 10, 12, 10)
        log_layout.addWidget(StrongBodyLabel("运行日志", log_card))
        self.logView = QPlainTextEdit(log_card)
        self.logView.setReadOnly(True)
        self.logView.setMaximumBlockCount(3000)
        self.logView.setStyleSheet("QPlainTextEdit { font-family: Consolas, monospace; }")
        log_layout.addWidget(self.logView)
        layout.addWidget(log_card, 1)

    def _add_command_button(self, row: QHBoxLayout, text: str, command_id: int, parent: QWidget) -> None:
        button = PrimaryPushButton(text, parent)
        button.clicked.connect(lambda: self.controller.send_command(command_id))
        row.addWidget(button)

    def _confirm_clear_data(self) -> None:
        answer = QMessageBox.question(
            self,
            "清空 SD 数据",
            "确认删除设备 SD 卡上的全部数据？该操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self.controller.send_command(CMD_CLEAR_DATA)
            notify(self, "已发送", "清空 SD 数据命令", "warning")

    def append_log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.logView.appendPlainText(f"[{stamp}] {message}")

    def _on_response(self, response: CommandResponse) -> None:
        row = self.responseTable.rowCount()
        self.responseTable.insertRow(row)
        name = COMMAND_NAMES.get(response.command_id, f"0x{response.command_id:02X}")
        status = "成功" if response.ok else f"失败({response.status})"
        cells = (
            datetime.now().strftime("%H:%M:%S"),
            name,
            status,
            response.message,
        )
        for column, text in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter if column in (0, 2) else Qt.AlignLeft | Qt.AlignVCenter)
            self.responseTable.setItem(row, column, item)
        self.responseTable.scrollToBottom()

    def _clear_all(self) -> None:
        self.logView.clear()
        self.responseTable.setRowCount(0)

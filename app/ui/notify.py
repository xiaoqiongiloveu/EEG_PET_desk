"""统一的浮动提示。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import InfoBar, InfoBarPosition

_LEVELS = {
    "success": InfoBar.success,
    "warning": InfoBar.warning,
    "error": InfoBar.error,
    "info": InfoBar.info,
}


def notify(parent: QWidget | None, title: str, content: str = "", level: str = "success") -> None:
    handler = _LEVELS.get(level, InfoBar.info)
    window = parent.window() if parent is not None else None
    handler(
        title,
        content,
        orient=Qt.Horizontal,
        isClosable=True,
        duration=2600,
        position=InfoBarPosition.TOP_RIGHT,
        parent=window,
    )

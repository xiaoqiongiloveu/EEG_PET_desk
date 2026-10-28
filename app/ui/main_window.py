"""主窗口：四个功能页面的容器。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication
from qfluentwidgets import FluentIcon, FluentWindow, NavigationItemPosition, setTheme, Theme

from .. import APP_NAME, APP_VERSION
from ..config import Settings, save_settings
from ..controller import AcquisitionController
from .pages.acquisition_page import AcquisitionPage
from .pages.log_page import LogPage
from .pages.replay_page import ReplayPage
from .pages.settings_page import SettingsPage


class MainWindow(FluentWindow):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        setTheme(Theme.LIGHT)
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self._apply_initial_size()

        self.controller = AcquisitionController(settings, self)
        self.acquisitionPage = AcquisitionPage(self.controller, settings, self)
        self.replayPage = ReplayPage(settings, self)
        self.logPage = LogPage(self.controller, self)
        self.settingsPage = SettingsPage(settings, self)

        self.addSubInterface(self.acquisitionPage, FluentIcon.HOME, "实时采集")
        self.addSubInterface(self.replayPage, FluentIcon.HISTORY, "回放分析")
        self.addSubInterface(self.logPage, FluentIcon.SEND, "命令与日志")
        self.addSubInterface(
            self.settingsPage,
            FluentIcon.SETTING,
            "设置",
            NavigationItemPosition.BOTTOM,
        )
        self.navigationInterface.setExpandWidth(190)
        self._move_to_center()

        self.settingsPage.settings_saved.connect(self._on_settings_saved)

    def _apply_initial_size(self) -> None:
        """初始窗口纵向放大一倍，并限制在当前屏幕可用区域内。"""

        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1500, 1900)
            return
        available = screen.availableGeometry()
        width = min(1500, max(available.width() - 40, 800))
        height = min(1900, max(available.height() - 40, 600))
        self.resize(width, height)

    def _move_to_center(self) -> None:
        screen = self.screen()
        if screen is None:
            return
        geometry = screen.availableGeometry()
        frame = self.frameGeometry()
        frame.moveCenter(geometry.center())
        self.move(frame.topLeft())

    def _on_settings_saved(self, settings: Settings) -> None:
        self.settings = settings
        self.acquisitionPage.apply_settings(settings)
        self.controller.set_rereference(settings.reref_mode, settings.reref_channel)
        self.replayPage.settings = settings

    def closeEvent(self, event: QCloseEvent) -> None:
        try:
            self.controller.shutdown()
            save_settings(self.settings)
        finally:
            super().closeEvent(event)

"""程序入口。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from app import APP_NAME, APP_VERSION
from app.config import load_settings
from app.ui.main_window import MainWindow

SPLASH_ASSET = "assets/splash.jpg"
SPLASH_SCALE = 0.5
SPLASH_MIN_MS = 2000


def _resource_path(relative: str) -> Path:
    """返回资源路径：PyInstaller 打包后优先用解包目录。"""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative


def main() -> int:
    settings = load_settings()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(f"{APP_NAME} v{APP_VERSION}")

    splash: QSplashScreen | None = None
    splash_path = _resource_path(SPLASH_ASSET)
    first_frame = QPixmap(str(splash_path)) if splash_path.is_file() else QPixmap()
    if not first_frame.isNull():
        size = first_frame.size()
        first_frame = first_frame.scaled(
            max(int(size.width() * SPLASH_SCALE), 1),
            max(int(size.height() * SPLASH_SCALE), 1),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        splash = QSplashScreen(first_frame)
        splash.show()
        app.processEvents()

    started = time.monotonic()
    window = MainWindow(settings)
    if splash is None:
        window.show()
    else:
        # 至少展示 SPLASH_MIN_MS，即使主窗口已经构造完成。
        remaining = max(0, SPLASH_MIN_MS - int((time.monotonic() - started) * 1000))

        def reveal() -> None:
            window.show()
            splash.finish(window)

        QTimer.singleShot(remaining, reveal)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

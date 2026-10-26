"""程序入口。"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app import APP_NAME, APP_VERSION
from app.config import load_settings
from app.ui.main_window import MainWindow


def main() -> int:
    settings = load_settings()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(f"{APP_NAME} v{APP_VERSION}")
    window = MainWindow(settings)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

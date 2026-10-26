"""采集工作线程的公共接口。

串口线程与模拟数据源都实现同一组信号和命令方法，界面层无需区分数据来源。
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal

from ..constants import DEFAULT_BAUDRATE, EEG_GAIN, SAMPLE_RATE


@dataclass
class StreamConfig:
    """建立连接与配置设备所需的参数。"""

    port: str = ""
    baudrate: int = DEFAULT_BAUDRATE
    sample_rate: int = SAMPLE_RATE
    gain: int = EEG_GAIN
    auto_start: bool = True
    # True 表示通过电脑端 WiFi 模块的虚拟串口透传，端口必须由用户显式选择。
    wifi: bool = False


class AcquisitionWorker(QThread):
    """数据源统一接口。"""

    connected = Signal(bool, str)
    # 串口打开后收到第一帧有效数据时发出，用于区分“端口已打开”和“设备已响应”。
    device_ready = Signal(str)
    samples_batch = Signal(object)  # SampleBatch
    impedance_received = Signal(object)  # np.ndarray (8,) kΩ
    status_received = Signal(object)  # DeviceStatus
    aux_received = Signal(object)  # AuxPayload
    stats_received = Signal(object)  # AcquisitionStats
    command_response = Signal(object)  # CommandResponse
    log_message = Signal(str)
    error_occurred = Signal(str)

    def request_stop(self) -> None:
        raise NotImplementedError

    def start_stream(self) -> None:
        raise NotImplementedError

    def stop_stream(self) -> None:
        raise NotImplementedError

    def start_impedance(self) -> None:
        raise NotImplementedError

    def stop_impedance(self) -> None:
        raise NotImplementedError

    def query_state(self) -> None:
        raise NotImplementedError

    def send_command(self, command_id: int, payload: bytes = b"") -> None:
        raise NotImplementedError

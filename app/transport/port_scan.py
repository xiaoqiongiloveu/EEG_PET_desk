"""串口枚举与设备自动识别。"""

from __future__ import annotations

from dataclasses import dataclass

from serial.tools import list_ports

from ..constants import DEVICE_PID, DEVICE_VID


@dataclass(frozen=True, slots=True)
class PortInfo:
    device: str
    description: str
    hwid: str
    is_target: bool


def _matches_target(hwid: str) -> bool:
    marker = f"VID:PID={DEVICE_VID:04X}:{DEVICE_PID:04X}"
    return marker in hwid.upper()


def list_serial_ports() -> list[PortInfo]:
    """枚举全部串口，并标记 STM32 CDC 设备。"""

    ports: list[PortInfo] = []
    for port in list_ports.comports():
        hwid = port.hwid or ""
        ports.append(
            PortInfo(
                device=port.device,
                description=port.description or "",
                hwid=hwid,
                is_target=_matches_target(hwid),
            )
        )
    ports.sort(key=lambda item: (not item.is_target, item.device))
    return ports


def find_target_port() -> str | None:
    """返回第一个 VID:PID=0483:5740 的串口；找不到时返回 None。"""

    for port in list_serial_ports():
        if port.is_target:
            return port.device
    return None


def resolve_port(preferred: str = "", strict: bool = False) -> str | None:
    """解析要打开的串口。

    preferred 存在时优先使用；strict=True 时不再回退到 STM32 CDC 自动识别，
    用于 WiFi 模块场景（端口必须由用户显式选择）。
    """

    ports = list_serial_ports()
    if preferred and any(port.device.upper() == preferred.upper() for port in ports):
        return preferred
    if strict:
        return None
    return next((port.device for port in ports if port.is_target), None)

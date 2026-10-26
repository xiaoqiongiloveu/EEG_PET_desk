"""连接层测试：串口解析、WiFi 透传模式与断线重连。"""

from __future__ import annotations

import struct
import unittest
from unittest import mock

import serial
from PySide6.QtCore import QObject, Signal

from app.config import Settings
from app.constants import (
    CMD_ENABLE_DEVICE,
    CMD_GET_STATE,
    CMD_SET_CONFIG,
    CMD_START_STREAM,
    CMD_STOP_STREAM,
    TYPE_EEG,
    WIFI_SAMPLE_RATE,
)
from app.controller import AcquisitionController
from app.protocol import FrameParser, pack_frame
from app.transport.base import StreamConfig
from app.transport.port_scan import PortInfo, resolve_port
from app.transport.serial_worker import SerialWorker


class FakeSerial:
    """可编排读取片段的串口替身：支持分片、粘包、异常与耗尽回调。"""

    def __init__(self, chunks, on_exhausted=None, raise_after: bool = False) -> None:
        self.chunks = [bytes(chunk) for chunk in chunks]
        self.on_exhausted = on_exhausted
        self.raise_after = raise_after
        self.writes: list[bytes] = []
        self.closed = False
        self.reset_calls = 0

    @property
    def in_waiting(self) -> int:
        return len(self.chunks[0]) if self.chunks else 0

    def read(self, size: int) -> bytes:
        if not self.chunks:
            if self.raise_after:
                raise serial.SerialException("链路已断开")
            if self.on_exhausted is not None:
                self.on_exhausted()
            return b""
        chunk = self.chunks[0]
        data, rest = chunk[:size], chunk[size:]
        if rest:
            self.chunks[0] = rest
        else:
            self.chunks.pop(0)
        return data

    def reset_input_buffer(self) -> None:
        self.reset_calls += 1

    def write(self, data: bytes) -> int:
        self.writes.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _eeg_frame(sequence: int, timestamp_ms: int) -> bytes:
    return pack_frame(TYPE_EEG, sequence, timestamp_ms, bytes(47))


def _command_ids(fake: FakeSerial) -> list[int]:
    # 命令帧：4 字节帧头 + TYPE + SEQ(2) + TS(4)，载荷从第 11 字节开始。
    return [frame[11] for frame in fake.writes]


class PortResolveTests(unittest.TestCase):
    def test_strict_mode_never_falls_back(self) -> None:
        ports = [PortInfo("COM9", "STM32 CDC", "USB VID:PID=0483:5740", True)]
        with mock.patch("app.transport.port_scan.list_serial_ports", return_value=ports):
            # WiFi 模式：用户选择的端口不在列表里时不做任何自动识别。
            self.assertIsNone(resolve_port("COM5", strict=True))
            # 用户明确选中的端口原样返回。
            self.assertEqual(resolve_port("com9", strict=True), "com9")
            # USB 模式保持原有回退行为。
            self.assertEqual(resolve_port("COM5", strict=False), "COM9")


class StreamParserTests(unittest.TestCase):
    def test_byte_by_byte_stream(self) -> None:
        raw = _eeg_frame(1, 10) + _eeg_frame(2, 20)
        parser = FrameParser()
        frames = []
        for index in range(len(raw)):
            frames.extend(parser.feed(raw[index : index + 1]))
        self.assertEqual([frame.sequence for frame in frames], [1, 2])

    def test_sticky_and_noisy_chunks(self) -> None:
        raw = b"\x00\x13" + _eeg_frame(7, 70) + _eeg_frame(8, 80) + b"\x99"
        parser = FrameParser()
        frames = parser.feed(raw[:20]) + parser.feed(raw[20:])
        self.assertEqual([frame.sequence for frame in frames], [7, 8])
        self.assertEqual(parser.crc_errors, 0)

    def test_noise_only_buffer_stays_small(self) -> None:
        parser = FrameParser()
        parser.feed(b"\x01" * 5000)
        self.assertLessEqual(len(parser.buffer), 1)


class SerialWorkerTests(unittest.TestCase):
    def _run_worker(self, config: StreamConfig, chunks, raise_after: bool = False):
        holder: dict = {}
        errors: list[str] = []
        events: list[tuple[bool, str]] = []
        ready: list[str] = []
        batches: list = []

        def on_exhausted() -> None:
            holder["worker"].request_stop()

        fake = FakeSerial(chunks, on_exhausted, raise_after)
        worker = SerialWorker(config)
        holder["worker"] = worker
        worker.error_occurred.connect(errors.append)
        worker.connected.connect(lambda ok, port: events.append((ok, port)))
        worker.device_ready.connect(ready.append)
        worker.samples_batch.connect(batches.append)

        with mock.patch(
            "app.transport.serial_worker.resolve_port", return_value=config.port
        ), mock.patch("app.transport.serial_worker.serial.Serial", return_value=fake):
            worker.run()
        return fake, errors, events, ready, batches

    def test_wifi_stream_opens_1m_8n1_and_reads_fragmented_frames(self) -> None:
        config = StreamConfig(
            port="COM7", baudrate=1_000_000, sample_rate=WIFI_SAMPLE_RATE, wifi=True
        )
        first = _eeg_frame(1, 10)
        second = _eeg_frame(2, 20)
        chunks = [first[:6], first[6:] + b"\x00\xff" + second, b"\x55"]
        fake, errors, events, ready, batches = self._run_worker(config, chunks)

        self.assertEqual(errors, [])
        self.assertEqual(events, [(True, "COM7"), (False, "")])
        self.assertEqual(ready, ["COM7"])
        self.assertEqual(sum(len(batch.values) for batch in batches), 2)

        self.assertEqual(fake.writes[0][11], CMD_ENABLE_DEVICE)
        self.assertEqual(
            _command_ids(fake),
            [
                CMD_ENABLE_DEVICE,
                CMD_SET_CONFIG,
                CMD_GET_STATE,
                CMD_START_STREAM,
                CMD_STOP_STREAM,
            ],
        )
        self.assertEqual(struct.unpack("<H", fake.writes[1][12:14])[0], WIFI_SAMPLE_RATE)

    def test_serial_open_parameters_are_8n1(self) -> None:
        config = StreamConfig(
            port="COM7", baudrate=1_000_000, sample_rate=WIFI_SAMPLE_RATE, wifi=True
        )
        captured: dict = {}
        fake = FakeSerial([_eeg_frame(1, 10)])
        worker = SerialWorker(config)

        def factory(**kwargs):
            captured.update(kwargs)
            return fake

        with mock.patch(
            "app.transport.serial_worker.resolve_port", return_value="COM7"
        ), mock.patch("app.transport.serial_worker.serial.Serial", side_effect=factory):
            worker._stop_requested = True
            worker.run()

        self.assertEqual(captured["port"], "COM7")
        self.assertEqual(captured["baudrate"], 1_000_000)
        self.assertEqual(captured["bytesize"], serial.EIGHTBITS)
        self.assertEqual(captured["parity"], serial.PARITY_NONE)
        self.assertEqual(captured["stopbits"], serial.STOPBITS_ONE)

    def test_no_device_response_reports_timeout_not_connected(self) -> None:
        config = StreamConfig(port="COM7", baudrate=1_000_000, wifi=True)
        with mock.patch("app.transport.serial_worker.WIFI_READY_TIMEOUT_S", 0.05):
            _fake, errors, events, ready, _batches = self._run_worker(config, [])
        self.assertEqual(ready, [])
        self.assertTrue(any("未收到设备有效数据" in message for message in errors))
        self.assertTrue(any("WiFi" in message for message in errors))
        self.assertEqual(events[-1], (False, ""))

    def test_disconnect_and_reconnect(self) -> None:
        config = StreamConfig(port="COM7", baudrate=1_000_000, wifi=True)
        _fake, errors, events, ready, _batches = self._run_worker(
            config, [_eeg_frame(1, 10)], raise_after=True
        )
        self.assertTrue(any("串口读写异常" in message for message in errors))
        self.assertEqual(ready, ["COM7"])
        self.assertEqual(events[-1], (False, ""))

        _fake2, errors2, events2, ready2, _b2 = self._run_worker(
            config, [_eeg_frame(5, 50)]
        )
        self.assertEqual(errors2, [])
        self.assertEqual(ready2, ["COM7"])
        self.assertEqual(events2, [(True, "COM7"), (False, "")])


class FakeWorker(QObject):
    connected = Signal(bool, str)
    device_ready = Signal(str)
    samples_batch = Signal(object)
    impedance_received = Signal(object)
    status_received = Signal(object)
    aux_received = Signal(object)
    stats_received = Signal(object)
    command_response = Signal(object)
    log_message = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, config: StreamConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.port = config.port
        self.stopped = False

    def start(self) -> None:
        pass

    def isRunning(self) -> bool:  # noqa: N802 - 与 QThread 接口一致
        return True

    def request_stop(self) -> None:
        self.stopped = True

    def apply_config(self, sample_rate: int, gain: int) -> None:
        self.config.sample_rate = int(sample_rate)
        self.config.gain = int(gain)

    def wait(self, msecs: int = 0) -> bool:
        return True


class ControllerModeTests(unittest.TestCase):
    def test_wifi_and_usb_modes_share_protocol_but_differ_in_config(self) -> None:
        settings = Settings()
        settings.sample_rate = 500
        controller = AcquisitionController(settings)

        with mock.patch("app.controller.SerialWorker", FakeWorker):
            controller.connect_serial("COM6", 1_000_000, wifi=True)
            wifi_worker = controller.worker
            self.assertEqual(controller.mode, "wifi")
            self.assertEqual(controller.active_baudrate, 1_000_000)
            self.assertTrue(wifi_worker.config.wifi)
            self.assertEqual(wifi_worker.config.sample_rate, WIFI_SAMPLE_RATE)
            self.assertEqual(settings.sample_rate, WIFI_SAMPLE_RATE)

            # 连接中再次下发配置时仍被限制在 250 SPS。
            controller.apply_config(1000, 24)
            self.assertEqual(settings.sample_rate, WIFI_SAMPLE_RATE)
            self.assertEqual(wifi_worker.config.sample_rate, WIFI_SAMPLE_RATE)

            settings.sample_rate = 1000
            controller.connect_serial("COM3", 460_800, wifi=False)
            self.assertEqual(controller.mode, "usb")
            self.assertEqual(controller.active_baudrate, 460_800)
            self.assertIsNot(controller.worker, wifi_worker)
            self.assertTrue(wifi_worker.stopped)
            self.assertFalse(controller.worker.config.wifi)
            self.assertEqual(controller.worker.config.sample_rate, 1000)

            controller.disconnect_device()
            self.assertIsNone(controller.worker)


if __name__ == "__main__":
    unittest.main()

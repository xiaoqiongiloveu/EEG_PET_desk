from __future__ import annotations

import struct
import unittest

import numpy as np

from app.constants import (
    TYPE_COMMAND,
    TYPE_COMMAND_RESPONSE,
)
from app.protocol import (
    FrameParser,
    build_command_frame,
    build_config_payload,
    crc16_ccitt,
    parse_aux_payload,
    parse_command_response,
    parse_eeg_payload,
    parse_impedance_payload,
    parse_status_payload,
    pack_frame,
)


class CrcTests(unittest.TestCase):
    def test_standard_vector(self) -> None:
        self.assertEqual(crc16_ccitt(b"123456789"), 0x29B1)


class FrameParserTests(unittest.TestCase):
    def test_command_roundtrip(self) -> None:
        raw = build_command_frame(0x03, 7, b"\x01\x02")
        frames = FrameParser().feed(raw)
        self.assertEqual(len(frames), 1)
        frame = frames[0]
        self.assertEqual(frame.packet_type, TYPE_COMMAND)
        self.assertEqual(frame.sequence, 7)
        self.assertEqual(frame.payload, b"\x03\x01\x02")

    def test_split_at_every_offset(self) -> None:
        raw = build_command_frame(0x04, 1)
        for cut in range(1, len(raw)):
            parser = FrameParser()
            self.assertEqual(parser.feed(raw[:cut]), [])
            frames = parser.feed(raw[cut:])
            self.assertEqual(len(frames), 1, f"cut={cut}")
            self.assertEqual(frames[0].payload, b"\x04")

    def test_garbage_prefix(self) -> None:
        raw = b"\x00\x11\x22" + build_command_frame(0x01, 2)
        frames = FrameParser().feed(raw)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].payload, b"\x01")

    def test_crc_error_then_valid_frame(self) -> None:
        first = bytearray(build_command_frame(0x01, 1))
        first[-1] ^= 0xFF
        parser = FrameParser()
        frames = parser.feed(bytes(first) + build_command_frame(0x02, 2))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].payload, b"\x02")
        self.assertEqual(parser.crc_errors, 1)

    def test_invalid_length_resync(self) -> None:
        bad = b"\x55\xAA\xFF\xFF" + b"\x00" * 8
        parser = FrameParser()
        self.assertEqual(parser.feed(bad), [])
        self.assertGreaterEqual(parser.malformed_frames, 1)

    def test_sticky_buffer_keeps_partial_sync(self) -> None:
        parser = FrameParser()
        self.assertEqual(parser.feed(b"\x55"), [])
        frames = parser.feed(b"\xAA" + build_command_frame(0x01, 1)[2:])
        self.assertEqual(len(frames), 1)


class PayloadTests(unittest.TestCase):
    def test_config_payload(self) -> None:
        payload = build_config_payload(500, 24, 0x0F, 1, 0, 5, 3600)
        self.assertEqual(len(payload), 11)
        self.assertEqual(struct.unpack("<H", payload[0:2])[0], 500)
        self.assertEqual(payload[2], 24)
        self.assertEqual(payload[3], 0x0F)
        self.assertEqual(payload[4], 1)
        self.assertEqual(payload[5], 0)
        self.assertEqual(payload[6], 5)
        self.assertEqual(struct.unpack("<I", payload[7:11])[0], 3600)

    def test_eeg_payload(self) -> None:
        raw_values = np.array([0, 1, -1, 8_388_607, -8_388_608, 1000, -1000, 123], dtype="<i4")
        payload = bytearray(47)
        payload[0] = 0x0F
        payload[1] = 0xF0
        payload[2] = 0x00
        payload[3:35] = raw_values.tobytes()
        struct.pack_into("<6h", payload, 35, 1000, -500, 0, 100, -50, 0)
        parsed = parse_eeg_payload(bytes(payload), 24)
        self.assertEqual(parsed.values_uv.shape, (8,))
        self.assertEqual(parsed.lead_off.sum(), 8)
        self.assertAlmostEqual(float(parsed.values_uv[0]), 0.0, places=5)
        expected_1 = 4_500_000.0 / (24 * 8_388_607.0)
        self.assertAlmostEqual(float(parsed.values_uv[1]), expected_1, places=5)
        self.assertAlmostEqual(float(parsed.acc_g[0]), 1.0, places=5)
        self.assertAlmostEqual(float(parsed.gyro_dps[0]), 10.0, places=5)

    def test_impedance_payload(self) -> None:
        payload = struct.pack("<8H", 1, 2, 3, 4, 5, 6, 7, 9999)
        parsed = parse_impedance_payload(payload)
        self.assertEqual(parsed.tolist(), [1, 2, 3, 4, 5, 6, 7, 9999])

    def test_aux_payload(self) -> None:
        payload = bytearray(16)
        struct.pack_into("<6h", payload, 0, 1000, 0, -1000, 100, 0, -100)
        struct.pack_into("<H", payload, 12, 415)
        payload[14] = 42
        parsed = parse_aux_payload(bytes(payload))
        self.assertAlmostEqual(float(parsed.acc_g[0]), 1.0)
        self.assertAlmostEqual(float(parsed.gyro_dps[2]), -10.0)
        self.assertAlmostEqual(parsed.battery_v, 4.15)
        self.assertEqual(parsed.motion_score, 42)

    def test_status_payload(self) -> None:
        payload = bytearray(11 + 6 + 12)
        payload[0] = 1
        payload[1] = 37
        payload[2] = 1
        struct.pack_into("<H", payload, 3, 402)
        payload[5] = 1
        struct.pack_into("<I", payload, 6, 12345)
        payload[10] = 3
        payload[11:17] = b"T1.0.1"
        payload[17] = 0
        payload[18:] = b"AIT2601X0456"
        status = parse_status_payload(bytes(payload))
        self.assertEqual(status.sd_usage_percent, 37)
        self.assertAlmostEqual(status.battery_v, 4.02)
        self.assertEqual(status.task_id, 12345)
        self.assertEqual(status.firmware, "T1.0.1")
        self.assertEqual(status.serial_no, "AIT2601X0456")

    def test_status_payload_link_stats_block(self) -> None:
        payload = bytearray(11 + 6 + 12)
        payload[0] = 1
        struct.pack_into("<H", payload, 3, 402)
        payload[11:17] = b"T4.1.0"
        payload[17] = 0
        payload[18:30] = b"AIT2601X0456"
        # 固件扩展块：SN 结束符 + 'N','D' + ads_drops + tx 溢出 + rx 溢出
        payload += b"\x00ND" + struct.pack("<III", 12, 3, 4)
        status = parse_status_payload(bytes(payload))
        self.assertEqual(status.serial_no, "AIT2601X0456")
        self.assertTrue(status.link_stats)
        self.assertEqual(status.ads_drops, 12)
        self.assertEqual(status.wireless_tx_overflow, 3)
        self.assertEqual(status.wireless_rx_overflow, 4)

    def test_command_response(self) -> None:
        raw = pack_frame(TYPE_COMMAND_RESPONSE, 0, 1, b"\x03\x00Config Updated")
        frame = FrameParser().feed(raw)[0]
        response = parse_command_response(frame.payload)
        self.assertTrue(response.ok)
        self.assertEqual(response.command_id, 0x03)
        self.assertEqual(response.message, "Config Updated")


if __name__ == "__main__":
    unittest.main()

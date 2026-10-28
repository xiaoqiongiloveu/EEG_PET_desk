"""会话记录与读取的单元测试。"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from app.core.models import AcquisitionStats
from app.storage import SessionReader, SessionRecorder, list_sessions


class StorageTest(unittest.TestCase):
    def test_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = [f"C{i}" for i in range(8)]
            recorder = SessionRecorder(root, {"sample_rate": 250, "gain": 24}, names)
            base = time.monotonic()
            values = np.random.default_rng(2).normal(0, 10, (250, 8)).astype(np.float32)
            recorder.append(values, base + np.arange(250) / 250)
            recorder.append_impedance(np.arange(8, dtype=float) + 1.0, base + 0.5)
            recorder.log_event("marker", "M1", label="M1")
            recorder.log_command("START_STREAM")
            time.sleep(0.1)
            path = recorder.stop()

            sessions = list_sessions(root)
            self.assertEqual(len(sessions), 1)
            reader = SessionReader(path)
            self.assertEqual(reader.sample_count, 250)
            data, timestamps = reader.read()
            self.assertEqual(data.shape, (250, 8))
            np.testing.assert_allclose(data, values)
            self.assertAlmostEqual(float(timestamps[0]), 0.0, places=3)
            self.assertAlmostEqual(float(timestamps[-1]), 0.996, places=3)

            block, _ = reader.read_time_range(0.0, 0.4)
            self.assertEqual(len(block), 100)

            events = reader.read_events()
            self.assertTrue(any(event["kind"] == "marker" for event in events))
            self.assertEqual(reader.read_impedance().shape, (1, 9))
            self.assertEqual(len(reader.read_commands()), 1)

    def test_loss_stats_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recorder = SessionRecorder(root, {"sample_rate": 250, "gain": 24}, ["C1"])
            recorder.append_stats(
                AcquisitionStats(
                    frames=250,
                    samples=250,
                    lost_by_seq=3,
                    lost_by_time=4,
                    loss_rate=3 / 253,
                    sequence_gaps=2,
                    outages=1,
                    sample_rate=249.8,
                    frame_rate=250.0,
                )
            )
            path = recorder.stop()

            header, row = (path / "stats.csv").read_text(encoding="utf-8").splitlines()
            self.assertTrue(header.startswith("time_s,frames,samples,lost_by_seq"))
            self.assertIn(",3,4,", row)

            summary = json.loads((path / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["loss_summary"]["lost_by_seq"], 3)
            self.assertEqual(summary["loss_summary"]["outages"], 1)

    def test_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(list_sessions(Path(temp)), [])


if __name__ == "__main__":
    unittest.main()

"""滤波、频谱与质量评估的单元测试。"""

from __future__ import annotations

import unittest

import numpy as np

from app.analysis.quality import artifact_mask, evaluate_channels
from app.analysis.spectrum import analyze_spectrum, spectral_edge_frequency
from app.analysis.stats import channel_statistics
from app.core.filters import RealtimeFilter, apply_rereference, offline_filter

FS = 250


def make_alpha(seconds: float = 10.0, channels: int = 8) -> np.ndarray:
    t = np.arange(0, seconds, 1 / FS)
    rng = np.random.default_rng(3)
    data = np.stack(
        [18 * np.sin(2 * np.pi * 10 * t + index * 0.2) + rng.normal(0, 2, t.size) for index in range(channels)],
        axis=1,
    )
    return data.astype(np.float32)


class SpectrumTest(unittest.TestCase):
    def test_alpha_peak_and_sef(self) -> None:
        result = analyze_spectrum(make_alpha(), FS)
        self.assertTrue(np.all((result.alpha_peak_hz >= 9.0) & (result.alpha_peak_hz <= 11.0)))
        self.assertTrue(np.all(result.sef95 > 9.0))
        self.assertEqual(result.psd.shape[0], 8)
        alpha = result.band_power["alpha"]
        theta = result.band_power["theta"]
        self.assertTrue(np.all(alpha > theta))

    def test_spectral_edge_frequency(self) -> None:
        freqs = np.linspace(0, 50, 501)
        psd = np.ones((1, 501))
        sef = spectral_edge_frequency(freqs, psd, ratio=0.5)
        self.assertAlmostEqual(float(sef[0]), 25.0, delta=1.0)


class QualityTest(unittest.TestCase):
    def test_flat_channel_is_flagged(self) -> None:
        data = make_alpha()
        data[:, 2] = 1e-5
        results = evaluate_channels(data, FS, [f"CH{i}" for i in range(8)])
        self.assertIn("平线", results[2].flags)
        self.assertLess(results[2].score, 60)

    def test_artifact_mask(self) -> None:
        data = make_alpha()
        data[100:110, 5] += 900
        mask = artifact_mask(data)
        self.assertTrue(mask[105])
        self.assertGreater(mask.sum(), 0)

    def test_statistics(self) -> None:
        stats = channel_statistics(make_alpha(2.0))
        self.assertEqual(len(stats), 8)
        self.assertGreater(stats[0].rms_uv, 5.0)


class FilterTest(unittest.TestCase):
    def test_realtime_filter_matches_offline(self) -> None:
        data = make_alpha(5.0)
        realtime = RealtimeFilter(FS, 8)
        realtime.configure(FS, 0.5, 45.0, 50, True)
        streamed = np.vstack(
            [realtime.process(data[start : start + 50]) for start in range(0, len(data), 50)]
        )
        offline = offline_filter(data, FS, 0.5, 45.0, 50)
        self.assertEqual(streamed.shape, offline.shape)
        self.assertTrue(np.isfinite(streamed).all())
        # 因果滤波与零相位滤波存在固定相位差，比较稳态幅度与主频即可。
        ratio = np.std(streamed[250:, 0]) / np.std(offline[250:, 0])
        self.assertGreater(ratio, 0.8)
        self.assertLess(ratio, 1.2)
        streamed_spectrum = analyze_spectrum(streamed[250:], FS)
        offline_spectrum = analyze_spectrum(offline[250:], FS)
        self.assertAlmostEqual(float(streamed_spectrum.alpha_peak_hz[0]), 10.0, delta=1.0)
        self.assertAlmostEqual(float(offline_spectrum.alpha_peak_hz[0]), 10.0, delta=1.0)

    def test_notch_attenuates_line_noise(self) -> None:
        t = np.arange(0, 10, 1 / FS)
        signal = np.stack([10 * np.sin(2 * np.pi * 50 * t)] * 8, axis=1)
        realtime = RealtimeFilter(FS, 8)
        realtime.configure(FS, 0.5, 45.0, 50, True)
        filtered = realtime.process(signal[125:])  # 跳过初始瞬态
        self.assertLess(np.std(filtered), 2.0)

    def test_rereference(self) -> None:
        data = make_alpha(2.0)
        car = apply_rereference(data, [f"CH{i}" for i in range(8)], mode="car")
        self.assertTrue(np.allclose(car.mean(axis=1), 0.0, atol=1e-3))
        referenced = apply_rereference(
            data, [f"CH{i}" for i in range(8)], mode="channel", reference_channel="CH0"
        )
        self.assertTrue(np.allclose(referenced[:, 0], data[:, 0] * 0.5, atol=1e-3))


if __name__ == "__main__":
    unittest.main()

"""链路丢包统计：序号口径与设备时间戳口径。

两种口径互补：
    - 序号口径：EEG 帧序号跳变，反映「设备已发出但主机没收到」的帧数；
    - 时间戳口径：相邻帧设备时间戳间隔大于采样周期，可发现设备采样端（如
      ADS1299 环形缓冲溢出）的丢包——这类丢包不会体现在序号上。

超出 ``MAX_OUTAGE_S`` 的跳变视为断流（设备重启、链路中断、重新同步），
单独计入 ``outages``，不计入丢样，避免把长时间断连误算成海量丢包。
"""

from __future__ import annotations

# 超过该时长的跳变按断流处理。
MAX_OUTAGE_S = 10.0
# 设备时间戳来自 HAL_GetTick()，分辨率 1 ms。采样周期短于该值时，
# 抖动与丢样无法区分，故只做序号口径统计。
MIN_TIMEBASE_PERIOD_MS = 4.0


class StreamLossTracker:
    """按帧推进的丢包计数器，只依赖帧序号与设备时间戳。"""

    def __init__(self, sample_rate: float = 250.0) -> None:
        self.sample_rate = float(sample_rate)
        self.lost_by_seq = 0
        self.lost_by_time = 0
        self.gaps = 0
        self.outages = 0
        self._last_seq: int | None = None
        self._last_ms: int | None = None

    # ------------------------------------------------------------------ 查询
    def loss_rate(self, received: int) -> float:
        total = int(received) + int(self.lost_by_seq)
        if total <= 0:
            return 0.0
        return self.lost_by_seq / total

    # ------------------------------------------------------------------ 更新
    def update(self, sequence: int, device_ms: int) -> None:
        seq = int(sequence) & 0xFFFF
        ts = int(device_ms) & 0xFFFFFFFF
        max_gap = max(1, int(self.sample_rate * MAX_OUTAGE_S))
        outage = False

        if self._last_seq is not None:
            gap = (seq - self._last_seq - 1) & 0xFFFF
            if gap:
                if gap <= max_gap:
                    self.lost_by_seq += gap
                    self.gaps += 1
                else:
                    # 序号跳变远超合理范围：设备重启或链路重新同步。
                    outage = True
        self._last_seq = seq

        if self._last_ms is not None and self._update_time(ts, max_gap):
            outage = True
        self._last_ms = ts
        if outage:
            self.outages += 1

    def _update_time(self, ts: int, max_gap: int) -> bool:
        """返回 True 表示该帧判定为断流而非普通丢样。"""

        assert self._last_ms is not None
        delta = (ts - self._last_ms) & 0xFFFFFFFF
        expected = 1000.0 / self.sample_rate if self.sample_rate > 0 else 0.0
        if delta >= 0x80000000:
            # 时间戳倒退：设备重启或计时器被重置。
            return True
        if expected <= 0.0 or expected < MIN_TIMEBASE_PERIOD_MS:
            return False
        if delta > max_gap * expected:
            return True
        # 半个采样周期以内的抖动不计丢样；多出的整数个周期才算丢样。
        missing = int((delta - 0.5 * expected) / expected)
        if missing > 0:
            self.lost_by_time += missing
        return False

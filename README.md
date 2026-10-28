# NiceDog 脑电采集上位机

Windows 图形化上位机，通过 USB CDC 虚拟串口接收 8 通道 ADS1299 脑电数据，支持实时显示、阻抗测试、数据记录与回放分析。

## 快速开始

```bat
run.bat
```

`run.bat` 默认使用 `D:\rest-eeg-mvp-v0.3.00\rest-eeg-mvp-v0.3.0\runtime\python.exe`；如需更换解释器，修改脚本第一行的 `PY` 变量，或直接运行：

```bat
python main.py
```

依赖见 `requirements.txt`。设备会按 `VID:PID=0483:5740` 自动识别串口。

## 使用流程

1. 连接设备 → 打开「实时采集」，选择串口后点击「连接」。
2. 连接后自动执行：使能设备 → 配置采样率/增益 → 查询状态 → 开始采集。
3. 「阻抗测试」进入阻抗模式，再次点击停止（固件命令 0x14，不会停止数据流）。
4. 「开始记录」将会话写入 `sessions/`；「打标记」写入事件，便于后续对齐。
5. 「回放分析」加载历史会话，可播放、倍速、拖动游标、读取任意时刻通道数值，并查看功率谱、统计与质量评估。

没有硬件时点击「模拟数据」，可完整体验采集、记录、回放与分析的整条链路。

## 数据格式

每帧遵循 55AA 二进制协议：

```
55 AA | LEN(2,LE) | TYPE(1) | SEQ(2,LE) | TS(4,LE,ms) | PAYLOAD | CRC16(2,LE)
```

`LEN = 7 + payload 长度`，CRC 为 CRC16-CCITT-FALSE，覆盖 `TYPE..PAYLOAD`。

| TYPE | 含义 | 载荷 |
| --- | --- | --- |
| 0x01 | EEG | 3B 导联脱落 + 8×int32 原始 ADC + 6×int16 IMU |
| 0x03 | 阻抗 | 8×uint16，单位 kΩ |
| 0x04 | 状态 | 设备类型/SD/电量/状态码/任务号/固件/SN |
| 0x05 | 辅助 | 加速度、角速度、电量、运动评分 |
| 0x11 | 命令响应 | cmd、status、文本 |

电压换算：`µV = raw × 4.5 / (gain × (2^23 − 1)) × 1e6`。

通道原始顺序为 `C1, C3, CZ, C2, C4, AF8, FPZ, AF7`，界面按 `C4, C2, CZ, C1, C3, AF7, FPZ, AF8` 显示。

## 会话目录

```
sessions/20260304_153000/
  session.json      元数据与统计
  samples.bin       float32 交错采样（原始未滤波）
  timestamps.bin    float64 相对时间戳（秒）
  stats.csv         每秒一行的丢包/速率统计（长时测试用）
  events.jsonl      标记与状态事件
  commands.log      下发命令
  impedance.tsv     阻抗记录
  notes.txt         备注
```

记录保存的是**原始采样**；显示滤波与重参考在采集链路上实时完成，回放时可随时调整滤波、重参考与窗长。

## 分析算法

- P0：实时 IIR 滤波（4 阶 Butterworth + 陷波）、环形缓冲、Welch PSD、频带功率、质量门控（平线/饱和/漂移/高频噪声/通道相关性）。
- P1：零相位滤波（回放）、CAR/指定通道重参考、SEF95、频谱重心、alpha 峰、稳健 z 分数伪迹标记、事件标记对齐。

## 自检与测试

```bat
python tools\diagnose_device.py COM3 -t 5
python -m unittest discover -s tests -v
```

`tools/diagnose_device.py` 会打印串口列表、设备状态、各类帧数量与 CRC 错误数，用于排查连接问题。

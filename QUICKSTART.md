# 🚀 Path 2 快速开始指南

## 完成情况概览

✅ **阶段1**: Motion Sequence Generator (运动序列生成器)  
✅ **阶段2**: LSTM-Based Multi-Object Tracker (LSTM跟踪器)

## 📁 文件清单

```
path2_output/
├── path2_stage1_2_implementation.py  # 核心实现代码
├── path2_visualization.py            # 可视化工具
├── PATH2_README.md                   # 详细文档
├── requirements.txt                  # Python依赖
└── QUICKSTART.md                     # 本文件
```

## 🔧 安装步骤

### 1. 安装依赖

```bash
# 推荐: 使用--break-system-packages flag
pip install pybullet numpy torch opencv-python matplotlib pyyaml --break-system-packages

# 或者使用requirements.txt
pip install -r requirements.txt --break-system-packages
```

### 2. 验证安装

```python
import pybullet as p
import torch
import numpy as np
print(f"PyBullet: {p.getVersionString()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
```

## 🎯 三种使用方式

### 方式1: 运行完整Pipeline (推荐新手)

```bash
python path2_stage1_2_implementation.py
```

**输出**:
- `./path2_output/stage1/motion_sequence.json` - 运动序列数据
- `./path2_output/stage2/best_lstm_tracker.pth` - 训练好的LSTM模型

**预期时间**: 5-10分钟

### 方式2: 分步骤运行

#### Step 1: 只生成运动序列

```python
from path2_stage1_2_implementation import run_stage1_demo

json_file = run_stage1_demo()
print(f"Generated: {json_file}")
```

#### Step 2: 只训练LSTM模型

```python
from path2_stage1_2_implementation import run_stage2_demo

json_file = "./path2_output/stage1/motion_sequence.json"
run_stage2_demo(json_file)
```

### 方式3: 自定义参数

```python
from path2_stage1_2_implementation import (
    MotionSequenceGenerator,
    MotionType,
    LSTMTracker,
    LSTMTrackerTrainer,
    TrackingDataset
)
from torch.utils.data import DataLoader

# === 阶段1: 自定义运动生成 ===
generator = MotionSequenceGenerator(
    scene_size=(15.0, 15.0),  # 更大的场景
    num_cameras=6,            # 更多相机
    fps=60,                   # 更高帧率
    output_dir="./my_custom_output"
)

# 添加复杂运动模式
generator.add_object(
    obj_id=1,
    start_pos=[2.0, 2.0, 0.5],
    motion_type=MotionType.CIRCULAR,
    center=[7.5, 7.5, 0.5],
    radius=3.0,
    angular_velocity=0.8
)

# 生成更长的序列
frame_data = generator.generate_sequence(
    duration=30.0,  # 30秒
    save_json=True
)

generator.cleanup()

# === 阶段2: 自定义LSTM训练 ===
dataset = TrackingDataset(
    json_file="./my_custom_output/motion_sequence.json",
    sequence_length=15,  # 更长的输入序列
    prediction_horizon=10  # 预测更多步
)

# 创建dataloader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 自定义模型架构
model = LSTMTracker(
    input_size=4,
    hidden_size=256,  # 更大的隐藏层
    num_layers=3,     # 更深的网络
    output_size=4
)

# 训练
trainer = LSTMTrackerTrainer(model=model, learning_rate=0.0005)
trainer.train(
    train_loader=train_loader,
    val_loader=train_loader,  # 简化示例
    num_epochs=100,
    save_path="./my_custom_model.pth"
)
```

## 📊 可视化结果

### 1. 可视化运动轨迹

```bash
python path2_visualization.py
```

或在Python中:

```python
from path2_visualization import visualize_motion_sequence

visualize_motion_sequence(
    json_file="./path2_output/stage1/motion_sequence.json",
    camera_id=0,  # 选择相机0
    save_gif=True
)
```

### 2. 分析LSTM预测

```python
from path2_visualization import analyze_lstm_predictions

analyze_lstm_predictions(
    model_path="./path2_output/stage2/best_lstm_tracker.pth",
    json_file="./path2_output/stage1/motion_sequence.json",
    num_samples=5  # 可视化5个样本
)
```

### 3. 计算性能指标

```python
from path2_visualization import compute_tracking_metrics

compute_tracking_metrics(
    model_path="./path2_output/stage2/best_lstm_tracker.pth",
    json_file="./path2_output/stage1/motion_sequence.json"
)
```

## 🎓 核心概念说明

### 阶段1: 运动序列生成器

**作用**: 生成多物体在多相机视角下的连续运动轨迹

**支持的运动类型**:
- `LINEAR`: 直线运动 (匀速直线)
- `CIRCULAR`: 圆周运动 (等角速度圆周)
- `RANDOM_WALK`: 随机游走 (布朗运动)
- `STATIONARY`: 静止

**输出数据**:
```json
{
  "frame": 120,
  "timestamp": 4.0,
  "camera_id": 0,
  "objects": [
    {
      "id": 1,
      "pos_3d": [1.2, 0.4, 0.5],
      "bbox": [120, 80, 245, 380],
      "occlusion": 0.15,
      "velocity": [0.5, 0.3, 0.0],
      "motion_type": "linear"
    }
  ]
}
```

### 阶段2: LSTM跟踪器

**作用**: 根据历史轨迹预测未来位置

**输入**: 过去10帧的bbox序列 `[x1, y1, x2, y2]`  
**输出**: 未来5帧的bbox预测

**模型结构**:
```
Input (batch, 10, 4)
    ↓
LSTM (hidden=128, layers=2)
    ↓
FC (128 → 64 → 4)
    ↓
Output (batch, 4)
```

**训练目标**: 最小化预测bbox与真实bbox的MSE

## 🐛 常见问题

### Q1: ImportError: No module named 'pybullet'

**解决方案**:
```bash
pip install pybullet --break-system-packages
```

### Q2: CUDA out of memory

**解决方案**:
- 减小batch_size: `batch_size=8` 或 `batch_size=4`
- 或使用CPU: 代码会自动检测并切换到CPU

### Q3: 生成的数据集为空 (len(dataset) == 0)

**原因**: 序列太短,不足以生成训练样本

**解决方案**:
- 增加duration: `duration=20.0` 或更长
- 减少sequence_length: `sequence_length=5`

### Q4: 训练loss不下降

**可能原因**:
1. 学习率太大: 尝试 `lr=0.0001`
2. 数据太少: 生成更多数据
3. 模型太简单: 增加hidden_size或num_layers

## 📈 性能基准

### 默认配置下的预期性能

**Stage 1 (运动生成)**:
- 生成速度: ~100-300 frames/sec
- 10秒序列 @ 30fps = 300 frames
- 生成时间: 1-3秒

**Stage 2 (LSTM训练)**:
- 数据集大小: ~50-500 samples
- 训练时间 (30 epochs):
  - CPU: ~2-5分钟
  - GPU: ~20-60秒
- 预期val_loss: < 100 (像素MSE)

## 🔄 与现有系统集成

### 与PyBullet多相机系统集成

```python
# 使用你现有的PyBullet场景
from path2_stage1_2_implementation import MotionSequenceGenerator

generator = MotionSequenceGenerator(...)
# ... 添加物体 ...
frame_data = generator.generate_sequence(...)

# 现在可以用这些数据训练YOLO或其他模型
```

### 与YOLO检测pipeline集成

```python
# LSTM预测 + YOLO检测 = 完整跟踪系统
# 1. YOLO检测当前帧
detections = yolo_model(frame)

# 2. LSTM预测下一帧位置
lstm_predictions = lstm_model.predict_sequence(history)

# 3. 数据关联 (Hungarian matching)
matched_tracks = hungarian_match(detections, lstm_predictions)
```

## 📚 进一步学习

### 推荐阅读
1. `PATH2_README.md` - 完整技术文档
2. `Two-Dimension_Dual-Leg_System_Plan_v2.md` - 整体架构
3. `future_plan_dual_path.md` - 未来规划

### 下一步
- [ ] **Stage 3**: ReID appearance variation simulation
- [ ] **Stage 4**: Complete integrated tracking system
- [ ] 集成Kalman滤波器
- [ ] 添加Hungarian数据关联
- [ ] 多相机融合

## ✅ 验证清单

运行完整pipeline后,检查:

- [ ] `./path2_output/stage1/motion_sequence.json` 文件存在
- [ ] JSON文件包含 >100 frames的数据
- [ ] `./path2_output/stage2/best_lstm_tracker.pth` 文件存在
- [ ] 模型文件大小 ~2-5 MB
- [ ] 训练loss收敛 (下降趋势)
- [ ] 可视化脚本运行成功

## 🎉 成功标志

如果你看到:

```
✅ Saved motion sequence to ./path2_output/stage1/motion_sequence.json
✅ Saved best model (val_loss: 0.XXXXX)
🎉 All stages completed successfully!
```

**恭喜!** Path 2的阶段1和2已成功实现!

## 📞 获取帮助

遇到问题?
1. 检查 `PATH2_README.md` 的Known Limitations部分
2. 查看代码注释
3. 检查输出的JSON文件格式是否正确

---

**Last Updated**: 2025-11-14  
**Version**: 1.0  
**Status**: ✅ Production Ready

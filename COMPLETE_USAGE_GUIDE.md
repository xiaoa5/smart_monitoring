# Smart Monitoring 完整使用指南

**版本:** v1.0
**更新日期:** 2025-11-18
**适用环境:** Google Colab / 本地环境

---

## 目录

1. [快速开始](#1-快速开始)
2. [环境配置](#2-环境配置)
3. [数据准备](#3-数据准备)
4. [模型训练](#4-模型训练)
5. [预测使用](#5-预测使用)
6. [高级功能](#6-高级功能)
7. [可视化](#7-可视化)
8. [故障排除](#8-故障排除)
9. [最佳实践](#9-最佳实践)
10. [API参考](#10-api参考)

---

## 1. 快速开始

### 1.1 5分钟快速体验（Colab）

```python
# 步骤1: 安装依赖
!pip install pybullet==3.2.7 numpy==2.1.1 torch opencv-python-headless matplotlib tqdm pyyaml

# 步骤2: 克隆/上传代码
# 上传以下文件到Colab:
# - path2_phase1_2_verified.py
# - path2_probabilistic_lstm.py
# - test_prediction.py

# 步骤3: 生成测试数据
!python path2_phase1_2_verified.py

# 步骤4: 运行测试
!python test_prediction.py
```

### 1.2 本地快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/xiaoa5/smart_monitoring.git
cd smart_monitoring

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements_verified.txt

# 4. 生成数据
python path2_phase1_2_verified.py

# 5. 运行测试
python test_prediction.py
```

---

## 2. 环境配置

### 2.1 系统要求

**最低配置:**
- Python 3.8+
- 4GB RAM
- 5GB 磁盘空间

**推荐配置:**
- Python 3.10+
- 8GB+ RAM
- NVIDIA GPU (4GB+ VRAM)
- 10GB 磁盘空间

### 2.2 依赖安装

#### 方法1: 使用requirements.txt

```bash
# Colab环境（推荐）
pip install -r requirements_verified.txt

# 本地环境（带GUI）
pip install -r requirements.txt
```

#### 方法2: 手动安装核心依赖

```bash
pip install pybullet==3.2.7
pip install numpy==2.1.1
pip install torch>=2.5.0
pip install opencv-python-headless==4.10.0.84  # Colab
# 或
pip install opencv-python>=4.5.0  # 本地
pip install matplotlib==3.9.2
pip install tqdm==4.66.5
pip install pyyaml==6.0.2
```

### 2.3 验证安装

```python
import sys
import torch
import numpy as np
import pybullet as p
import cv2
import matplotlib

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("NumPy:", np.__version__)
print("PyBullet:", p.getConnectionInfo())
print("OpenCV:", cv2.__version__)
print("Matplotlib:", matplotlib.__version__)
```

**预期输出:**
```
Python: 3.11.x
PyTorch: 2.5.x
CUDA available: True
NumPy: 2.1.1
PyBullet: ...
OpenCV: 4.10.0
Matplotlib: 3.9.2
```

---

## 3. 数据准备

### 3.1 使用PyBullet生成数据

**基础用法:**

```python
# 运行数据生成脚本
!python path2_phase1_2_verified.py
```

**生成的数据结构:**
```
output/
├── data/
│   ├── cam_0.json    # 相机0的观测数据
│   ├── cam_1.json    # 相机1的观测数据
│   ├── cam_2.json    # 相机2的观测数据
│   └── cam_3.json    # 相机3的观测数据
└── visualizations/   # 可视化图片（可选）
```

**数据格式:**
```json
[
  {
    "frame": 0,
    "timestamp": 0.0,
    "objects": [
      {
        "id": 0,
        "pos_3d": [1.5, 0.0, 0.5],
        "bbox": [0.6, 0.5, 0.2, 0.3],
        "visible": true
      }
    ]
  }
]
```

### 3.2 使用自定义数据

如果你有自己的数据，需要转换为以下格式：

```python
import json
import numpy as np

# 你的数据
frames = []

for frame_idx in range(num_frames):
    frame_data = {
        "frame": frame_idx,
        "timestamp": frame_idx * 0.1,  # 100ms间隔
        "objects": []
    }

    for obj_id in range(num_objects):
        obj_data = {
            "id": obj_id,
            "pos_3d": [x, y, z],  # 3D位置
            "bbox": [cx, cy, w, h],  # YOLO格式bbox
            "visible": True
        }
        frame_data["objects"].append(obj_data)

    frames.append(frame_data)

# 保存
with open('output/data/cam_0.json', 'w') as f:
    json.dump(frames, f, indent=2)
```

### 3.3 数据验证

```python
import json
import os

def verify_data(data_dir='output/data'):
    """验证数据完整性"""

    # 检查目录
    if not os.path.exists(data_dir):
        print(f"❌ 目录不存在: {data_dir}")
        return False

    # 检查JSON文件
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    print(f"✓ 找到 {len(json_files)} 个JSON文件")

    # 验证每个文件
    for json_file in json_files:
        filepath = os.path.join(data_dir, json_file)
        with open(filepath, 'r') as f:
            data = json.load(f)

        print(f"\n{json_file}:")
        print(f"  帧数: {len(data)}")
        print(f"  对象数: {len(data[0]['objects'])}")

        # 检查数据完整性
        for frame in data[:3]:  # 检查前3帧
            assert 'frame' in frame
            assert 'timestamp' in frame
            assert 'objects' in frame

            for obj in frame['objects']:
                assert 'id' in obj
                assert 'pos_3d' in obj
                assert 'bbox' in obj
                assert len(obj['pos_3d']) == 3
                assert len(obj['bbox']) == 4

    print("\n✅ 数据验证通过!")
    return True

# 运行验证
verify_data()
```

---

## 4. 模型训练

### 4.1 基础训练

**完整训练脚本:**

```python
import torch
from torch.utils.data import DataLoader, random_split
from path2_probabilistic_lstm import (
    LSTMConfig,
    ProbabilisticLSTMTracker,
    MultiCameraDataset,
    ProbabilisticTrainer
)

# ============================================================================
# 1. 配置
# ============================================================================

config = LSTMConfig(
    # 模型架构
    input_dim=4,
    hidden_dim=128,
    num_layers=2,
    output_dim=3,

    # 注意力机制
    attention_heads=4,
    attention_dim=64,

    # 训练参数
    learning_rate=1e-3,
    batch_size=32,
    num_epochs=50,
    weight_decay=1e-5,

    # 数据参数
    seq_len=10,
    max_cameras=4
)

# 设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# ============================================================================
# 2. 加载数据
# ============================================================================

# 创建数据集
dataset = MultiCameraDataset(
    json_dir='output/data',
    seq_len=config.seq_len,
    max_cameras=config.max_cameras,
    add_noise=True,       # 数据增强
    noise_std=0.02,       # 噪声强度
    missing_prob=0.1      # 缺失概率
)

print(f"✓ 加载了 {len(dataset)} 个序列")

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"训练集: {len(train_dataset)} 个序列")
print(f"验证集: {len(val_dataset)} 个序列")

# 创建DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=0  # Colab设为0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=0
)

# ============================================================================
# 3. 创建模型
# ============================================================================

model = ProbabilisticLSTMTracker(config)

# 统计参数
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n✓ 模型创建成功")
print(f"  参数量: {num_params:,}")

# ============================================================================
# 4. 训练
# ============================================================================

trainer = ProbabilisticTrainer(model, config, device)

print(f"\n{'='*60}")
print("开始训练")
print(f"{'='*60}")

history = trainer.train(
    train_loader,
    val_loader,
    num_epochs=config.num_epochs
)

print(f"\n{'='*60}")
print("训练完成!")
print(f"{'='*60}")

# ============================================================================
# 5. 保存模型
# ============================================================================

# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'history': history
}, 'model_checkpoint.pth')

print("\n✓ 模型已保存到: model_checkpoint.pth")
```

### 4.2 监控训练

**训练过程输出:**
```
============================================================
Epoch 1/50
============================================================
Training: 100%|██████████| 27/27 [00:05<00:00,  5.23it/s]

Train Loss: 0.534821
Val Loss:   0.512345
Val MAE:    0.087654
Val Std:    0.045123

✓ New best model (Val Loss: 0.512345)
```

**保存训练历史:**

```python
import matplotlib.pyplot as plt

# 绘制损失曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_losses'], label='Train')
plt.plot(history['val_losses'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['val_losses'])
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Performance')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300)
plt.show()
```

### 4.3 加载预训练模型

```python
# 加载检查点
checkpoint = torch.load('model_checkpoint.pth', map_location=device)

# 恢复配置
config = checkpoint['config']

# 创建模型
model = ProbabilisticLSTMTracker(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print("✓ 模型加载成功")
```

---

## 5. 预测使用

### 5.1 基础预测

```python
import torch
import numpy as np
from path2_probabilistic_lstm import ProbabilisticLSTMTracker, MultiCameraDataset

# 加载模型和数据
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ...  # 加载模型（见4.3）
val_dataset = ...  # 加载数据（见4.1）

# ============================================================================
# 获取测试样本
# ============================================================================

sample = val_dataset[0]

# 准备输入（添加batch维度）
bbox_seq = sample['bbox_seq'].unsqueeze(0).to(device)      # [1, 10, 4, 4]
camera_ids = sample['camera_ids'].unsqueeze(0).to(device)  # [1, 10, 4]
mask = sample['mask'].unsqueeze(0).to(device)              # [1, 10, 4]
ground_truth = sample['pos_3d_seq'].cpu().numpy()          # [10, 3]

# ============================================================================
# 预测
# ============================================================================

mean, std = model.predict_distribution(bbox_seq, camera_ids, mask)

# mean: [10, 3] - 预测的3D位置
# std:  [10, 3] - 预测的不确定性

# ============================================================================
# 查看结果
# ============================================================================

print("最后一个时间步:")
print(f"  预测位置: {mean[-1]}")
print(f"  不确定性: {std[-1]}")
print(f"  真实位置: {ground_truth[-1]}")
print(f"  预测误差: {np.abs(mean[-1] - ground_truth[-1])}")

# 95% 置信区间
lower = mean[-1] - 1.96 * std[-1]
upper = mean[-1] + 1.96 * std[-1]
print(f"\n95% 置信区间:")
print(f"  下界: {lower}")
print(f"  上界: {upper}")
print(f"  包含真值: {np.all((ground_truth[-1] >= lower) & (ground_truth[-1] <= upper))}")
```

### 5.2 批量预测

```python
# 批量处理多个样本
results = []

for i in range(min(10, len(val_dataset))):
    sample = val_dataset[i]

    bbox_seq = sample['bbox_seq'].unsqueeze(0).to(device)
    camera_ids = sample['camera_ids'].unsqueeze(0).to(device)
    mask = sample['mask'].unsqueeze(0).to(device)
    ground_truth = sample['pos_3d_seq'].cpu().numpy()

    # 预测
    mean, std = model.predict_distribution(bbox_seq, camera_ids, mask)

    # 计算误差
    mae = np.abs(mean - ground_truth).mean()

    results.append({
        'sample_id': i,
        'mae': mae,
        'avg_uncertainty': std.mean()
    })

    print(f"样本 {i}: MAE={mae:.6f}m, 不确定性={std.mean():.6f}m")

# 统计
maes = [r['mae'] for r in results]
print(f"\n总体统计:")
print(f"  平均MAE: {np.mean(maes):.6f}m")
print(f"  MAE标准差: {np.std(maes):.6f}m")
```

### 5.3 不同相机数量的对比

```python
# 对比不同相机数量下的预测不确定性

sample = val_dataset[0]
bbox_seq_full = sample['bbox_seq'].unsqueeze(0).to(device)
camera_ids_full = sample['camera_ids'].unsqueeze(0).to(device)
mask_full = sample['mask'].unsqueeze(0).to(device)

# 测试不同相机数量
camera_counts = [1, 2, 3, 4]
uncertainties = []

for num_cams in camera_counts:
    # 创建mask（只使用前num_cams个相机）
    mask_test = mask_full.clone()
    mask_test[:, :, num_cams:] = True  # 屏蔽多余的相机

    # 预测
    mean, std = model.predict_distribution(
        bbox_seq_full, camera_ids_full, mask_test
    )

    avg_uncertainty = std.mean().item()
    uncertainties.append(avg_uncertainty)

    print(f"{num_cams} 个相机: 平均不确定性 = {avg_uncertainty:.6f}m")

# 可视化
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(camera_counts, uncertainties, 'o-', linewidth=2, markersize=8)
plt.xlabel('相机数量')
plt.ylabel('平均不确定性 (m)')
plt.title('相机数量 vs 预测不确定性')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 6. 高级功能

### 6.1 集成约束系统

```python
from path2_constraints import CircleConstraint, GaussianDistribution
import numpy as np

# ============================================================================
# 1. LSTM预测
# ============================================================================

sample = val_dataset[0]
bbox_seq = sample['bbox_seq'].unsqueeze(0).to(device)
camera_ids = sample['camera_ids'].unsqueeze(0).to(device)
mask = sample['mask'].unsqueeze(0).to(device)

# 获取LSTM预测
mean, logvar = model(bbox_seq, camera_ids, mask)

# 转换为numpy（最后一个时间步）
lstm_mean = mean[0, -1].cpu().numpy()
lstm_std = torch.exp(0.5 * logvar[0, -1]).cpu().numpy()
lstm_cov = np.diag(lstm_std**2)

# 创建高斯分布
prior = GaussianDistribution(lstm_mean, lstm_cov)

print("LSTM预测:")
print(f"  均值: {prior.mean}")
print(f"  标准差: {prior.std}")

# ============================================================================
# 2. 应用约束
# ============================================================================

# 定义圆形约束
circle = CircleConstraint(
    center=[0, 0, 0.5],  # 圆心
    radius=1.5,           # 半径
    normal=[0, 0, 1]      # 法向量（Z轴）
)

# 贝叶斯更新
posterior = circle.constrain(
    prior,
    constraint_std_radial=0.01  # 径向约束强度
)

print("\n约束后预测:")
print(f"  均值: {posterior.mean}")
print(f"  标准差: {posterior.std}")

# ============================================================================
# 3. 对比分析
# ============================================================================

print("\n改进效果:")
print(f"  不确定性降低: {(1 - posterior.std.mean()/prior.std.mean())*100:.1f}%")

# 可视化
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 5))

# 3D可视化
ax1 = fig.add_subplot(121, projection='3d')

# 绘制圆形约束
theta = np.linspace(0, 2*np.pi, 100)
circle_x = circle.center[0] + circle.radius * np.cos(theta)
circle_y = circle.center[1] + circle.radius * np.sin(theta)
circle_z = np.full_like(circle_x, circle.center[2])
ax1.plot(circle_x, circle_y, circle_z, 'r-', linewidth=2, label='Constraint')

# 绘制预测
ax1.scatter(*prior.mean, c='blue', s=100, label='LSTM', marker='o')
ax1.scatter(*posterior.mean, c='green', s=100, label='Constrained', marker='^')

ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.legend()
ax1.set_title('3D Prediction with Constraint')

# 不确定性对比
ax2 = fig.add_subplot(122)
dims = ['X', 'Y', 'Z']
x = np.arange(len(dims))
width = 0.35

ax2.bar(x - width/2, prior.std, width, label='LSTM', color='blue', alpha=0.7)
ax2.bar(x + width/2, posterior.std, width, label='Constrained', color='green', alpha=0.7)

ax2.set_ylabel('不确定性 (m)')
ax2.set_title('不确定性对比')
ax2.set_xticks(x)
ax2.set_xticklabels(dims)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 6.2 使用集成系统

```python
from path2_integrated import IntegratedConfig, IntegratedTracker

# ============================================================================
# 配置
# ============================================================================

config = IntegratedConfig(
    # LSTM配置
    seq_len=10,
    max_cameras=4,
    hidden_dim=128,

    # 约束配置
    use_constraints=True,
    constraint_strength=0.01,

    # 训练配置
    learning_rate=1e-3,
    batch_size=32,
    num_epochs=50
)

# ============================================================================
# 创建集成跟踪器
# ============================================================================

tracker = IntegratedTracker(config, device=device)

# ============================================================================
# 预测（自动应用约束）
# ============================================================================

sample = val_dataset[0]
bbox_seq = sample['bbox_seq'].unsqueeze(0).to(device)
camera_ids = sample['camera_ids'].unsqueeze(0).to(device)
mask = sample['mask'].unsqueeze(0).to(device)

# 不使用约束
mean_lstm, std_lstm = tracker.predict(
    bbox_seq, camera_ids, mask,
    use_constraint=False
)

# 使用约束
mean_constrained, std_constrained = tracker.predict(
    bbox_seq, camera_ids, mask,
    use_constraint=True
)

print("结果对比:")
print(f"LSTM不确定性:        {std_lstm.mean():.6f}m")
print(f"约束后不确定性:      {std_constrained.mean():.6f}m")
print(f"不确定性降低:        {(1 - std_constrained.mean()/std_lstm.mean())*100:.1f}%")
```

---

## 7. 可视化

### 7.1 轨迹可视化

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_trajectory(mean, std, ground_truth=None, title="3D Trajectory"):
    """可视化3D轨迹和不确定性"""

    fig = plt.figure(figsize=(15, 10))

    # ========================================================================
    # 1. 3D轨迹
    # ========================================================================
    ax1 = fig.add_subplot(221, projection='3d')

    # 预测轨迹
    ax1.plot(mean[:, 0], mean[:, 1], mean[:, 2],
             'b-', linewidth=2, label='Predicted', alpha=0.8)

    # 起点和终点
    ax1.scatter(*mean[0], c='green', s=100, marker='o', label='Start')
    ax1.scatter(*mean[-1], c='red', s=100, marker='s', label='End')

    # 真实轨迹
    if ground_truth is not None:
        ax1.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
                'g--', linewidth=2, label='Ground Truth', alpha=0.6)

    # 不确定性（每隔几个点显示）
    step = max(1, len(mean) // 10)
    for i in range(0, len(mean), step):
        # X方向误差棒
        ax1.plot([mean[i, 0] - std[i, 0], mean[i, 0] + std[i, 0]],
                [mean[i, 1], mean[i, 1]],
                [mean[i, 2], mean[i, 2]], 'r-', alpha=0.3, linewidth=1)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ========================================================================
    # 2. XY平面投影
    # ========================================================================
    ax2 = fig.add_subplot(222)
    ax2.plot(mean[:, 0], mean[:, 1], 'b-', linewidth=2, label='Predicted')

    if ground_truth is not None:
        ax2.plot(ground_truth[:, 0], ground_truth[:, 1],
                'g--', linewidth=2, label='Ground Truth', alpha=0.6)

    ax2.scatter(mean[0, 0], mean[0, 1], c='green', s=100, marker='o')
    ax2.scatter(mean[-1, 0], mean[-1, 1], c='red', s=100, marker='s')

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Plane Projection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # ========================================================================
    # 3. 位置误差（如果有真实值）
    # ========================================================================
    ax3 = fig.add_subplot(223)
    time_steps = np.arange(len(mean))

    if ground_truth is not None:
        error = np.linalg.norm(mean - ground_truth, axis=1)
        ax3.plot(time_steps, error, 'r-', linewidth=2, label='Position Error')
        ax3.fill_between(time_steps, 0, error, alpha=0.3, color='red')
        ax3.axhline(y=error.mean(), color='k', linestyle='--',
                   label=f'Mean: {error.mean():.4f}m')

    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Error (m)')
    ax3.set_title('Prediction Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ========================================================================
    # 4. 不确定性随时间变化
    # ========================================================================
    ax4 = fig.add_subplot(224)
    uncertainty = np.linalg.norm(std, axis=1)

    ax4.plot(time_steps, uncertainty, 'b-', linewidth=2)
    ax4.fill_between(time_steps, 0, uncertainty, alpha=0.3, color='blue')
    ax4.axhline(y=uncertainty.mean(), color='k', linestyle='--',
               label=f'Mean: {uncertainty.mean():.4f}m')

    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Uncertainty (m)')
    ax4.set_title('Prediction Uncertainty')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig

# 使用示例
fig = visualize_trajectory(mean, std, ground_truth,
                          title="Probabilistic LSTM Prediction")
plt.savefig('trajectory_visualization.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 7.2 不确定性热图

```python
def plot_uncertainty_heatmap(mean, std, ground_truth=None):
    """绘制不确定性热图"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    dims = ['X', 'Y', 'Z']
    time_steps = np.arange(len(mean))

    # X, Y, Z分量的不确定性
    for i, (ax, dim) in enumerate(zip(axes.flat[:3], dims)):
        # 预测均值
        ax.plot(time_steps, mean[:, i], 'b-', linewidth=2, label='Mean')

        # 不确定性带（±1σ, ±2σ）
        ax.fill_between(time_steps,
                        mean[:, i] - std[:, i],
                        mean[:, i] + std[:, i],
                        alpha=0.3, color='blue', label='±1σ (68%)')
        ax.fill_between(time_steps,
                        mean[:, i] - 2*std[:, i],
                        mean[:, i] + 2*std[:, i],
                        alpha=0.15, color='blue', label='±2σ (95%)')

        # 真实值
        if ground_truth is not None:
            ax.plot(time_steps, ground_truth[:, i],
                   'g--', linewidth=2, label='Ground Truth')

        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'{dim} Position (m)')
        ax.set_title(f'{dim} Component with Uncertainty')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 总体不确定性
    ax = axes.flat[3]
    uncertainty_magnitude = np.linalg.norm(std, axis=1)

    im = ax.scatter(time_steps, uncertainty_magnitude,
                   c=uncertainty_magnitude, cmap='hot',
                   s=100, edgecolors='black', linewidth=0.5)
    ax.plot(time_steps, uncertainty_magnitude, 'k-', alpha=0.3)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Uncertainty Magnitude (m)')
    ax.set_title('Overall Uncertainty')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Uncertainty (m)')

    plt.suptitle('Prediction Uncertainty Analysis',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig

# 使用
fig = plot_uncertainty_heatmap(mean, std, ground_truth)
plt.savefig('uncertainty_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 8. 故障排除

### 8.1 常见错误

#### 错误1: `NameError: name 'bbox_seq' is not defined`

**原因:** 没有从数据集获取数据

**解决:**
```python
# 正确做法
sample = val_dataset[0]
bbox_seq = sample['bbox_seq'].unsqueeze(0).to(device)
camera_ids = sample['camera_ids'].unsqueeze(0).to(device)
mask = sample['mask'].unsqueeze(0).to(device)
```

#### 错误2: `RuntimeError: expected mat1 and mat2 to have the same dtype`

**原因:** 数据类型不匹配（已在最新版本修复）

**解决:** 更新到最新版本，或手动添加类型转换：
```python
bbox_seq = torch.from_numpy(bbox_seq).float()
```

#### 错误3: `FileNotFoundError: output/data not found`

**原因:** 未生成数据

**解决:**
```bash
python path2_phase1_2_verified.py
```

#### 错误4: `CUDA out of memory`

**原因:** GPU内存不足

**解决:**
```python
# 方法1: 减小batch size
config.batch_size = 16  # 或更小

# 方法2: 使用CPU
device = 'cpu'

# 方法3: 清理缓存
torch.cuda.empty_cache()
```

#### 错误5: `IndexError: list index out of range`

**原因:** 数据集为空或索引错误

**解决:**
```python
# 检查数据集长度
print(f"数据集大小: {len(dataset)}")

# 确保索引有效
if len(dataset) > 0:
    sample = dataset[0]
else:
    print("数据集为空，请先生成数据")
```

### 8.2 性能优化

#### 优化1: 使用GPU加速

```python
# 确保使用GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# 使用混合精度训练（PyTorch 1.6+）
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        output = model(batch)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### 优化2: 数据加载加速

```python
# 增加num_workers（本地环境）
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,      # 根据CPU核心数调整
    pin_memory=True     # GPU加速
)
```

#### 优化3: 模型优化

```python
# 使用TorchScript编译
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# 加载使用
scripted_model = torch.jit.load("model_scripted.pt")
```

### 8.3 调试技巧

```python
# 1. 打印张量形状
print(f"bbox_seq shape: {bbox_seq.shape}")
print(f"Expected: [batch, seq_len, num_cameras, 4]")

# 2. 检查数据范围
print(f"bbox_seq range: [{bbox_seq.min():.3f}, {bbox_seq.max():.3f}]")
print(f"Expected: [0, 1] for normalized coords")

# 3. 可视化中间结果
def debug_forward(model, bbox_seq, camera_ids, mask):
    # 逐层检查
    fusion_output = model.camera_fusion(bbox_seq[:, 0], camera_ids[:, 0], mask[:, 0])
    print(f"Fusion output shape: {fusion_output.shape}")

    lstm_output, _ = model.lstm(fusion_output.unsqueeze(1))
    print(f"LSTM output shape: {lstm_output.shape}")

    return fusion_output, lstm_output

# 4. 梯度检查
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm():.6f}")
```

---

## 9. 最佳实践

### 9.1 数据处理

**建议:**
1. ✅ 使用数据增强（噪声、缺失）提高鲁棒性
2. ✅ 验证数据完整性和格式
3. ✅ 归一化bbox坐标（YOLO格式）
4. ✅ 使用适当的train/val分割（80/20）

**示例:**
```python
# 好的实践
dataset = MultiCameraDataset(
    json_dir='output/data',
    seq_len=10,
    add_noise=True,        # 训练时启用
    noise_std=0.02,
    missing_prob=0.1
)

# 固定随机种子保证可重复性
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
```

### 9.2 模型训练

**建议:**
1. ✅ 使用学习率调度
2. ✅ 梯度裁剪防止爆炸
3. ✅ 早停避免过拟合
4. ✅ 保存最佳模型

**示例:**
```python
# 早停实现
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 停止训练
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

# 使用
early_stopping = EarlyStopping(patience=10)

for epoch in range(num_epochs):
    train_loss = trainer.train_epoch(train_loader)
    val_loss = trainer.validate(val_loader)

    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

### 9.3 预测使用

**建议:**
1. ✅ 批量处理提高效率
2. ✅ 使用torch.no_grad()节省内存
3. ✅ 合理解释不确定性
4. ✅ 验证预测结果

**示例:**
```python
@torch.no_grad()
def batch_predict(model, samples, device='cuda'):
    """批量预测"""
    model.eval()
    results = []

    # 批量处理
    batch_size = 32
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]

        # 处理batch...
        bbox_seq = torch.stack([s['bbox_seq'] for s in batch]).to(device)
        # ...

        mean, logvar = model(bbox_seq, camera_ids, mask)
        std = torch.exp(0.5 * logvar)

        results.extend(zip(mean.cpu().numpy(), std.cpu().numpy()))

    return results
```

### 9.4 代码组织

**建议:**
1. ✅ 使用配置类管理参数
2. ✅ 模块化代码
3. ✅ 添加类型注解
4. ✅ 编写docstring

**示例:**
```python
from dataclasses import dataclass
from typing import Optional, Tuple
import torch

@dataclass
class ExperimentConfig:
    """实验配置"""
    # 数据
    data_dir: str = 'output/data'
    seq_len: int = 10

    # 模型
    hidden_dim: int = 128
    num_layers: int = 2

    # 训练
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 50

    # 设备
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_experiment(config: ExperimentConfig) -> Dict[str, any]:
    """
    运行完整实验流程

    Args:
        config: 实验配置

    Returns:
        包含结果的字典
    """
    # 实现...
    pass
```

---

## 10. API参考

### 10.1 核心类

#### ProbabilisticLSTMTracker

```python
class ProbabilisticLSTMTracker(nn.Module):
    """概率LSTM跟踪器"""

    def __init__(self, config: LSTMConfig):
        """
        Args:
            config: LSTM配置
        """
        pass

    def forward(
        self,
        bbox_seq: torch.Tensor,
        camera_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            bbox_seq: [batch, seq_len, num_cameras, 4]
            camera_ids: [batch, seq_len, num_cameras]
            mask: [batch, seq_len, num_cameras]

        Returns:
            mean: [batch, seq_len, 3]
            logvar: [batch, seq_len, 3]
        """
        pass

    def predict_distribution(
        self,
        bbox_seq: torch.Tensor,
        camera_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测分布

        Returns:
            mean: [seq_len, 3]
            std: [seq_len, 3]
        """
        pass
```

#### MultiCameraDataset

```python
class MultiCameraDataset(Dataset):
    """多相机数据集"""

    def __init__(
        self,
        json_dir: str,
        seq_len: int = 10,
        max_cameras: int = 4,
        add_noise: bool = False,
        noise_std: float = 0.02,
        missing_prob: float = 0.1
    ):
        """
        Args:
            json_dir: JSON文件目录
            seq_len: 序列长度
            max_cameras: 最大相机数
            add_noise: 是否添加噪声
            noise_std: 噪声标准差
            missing_prob: 缺失概率
        """
        pass

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                'bbox_seq': [seq_len, max_cameras, 4],
                'pos_3d_seq': [seq_len, 3],
                'camera_ids': [seq_len, max_cameras],
                'mask': [seq_len, max_cameras]
            }
        """
        pass
```

### 10.2 约束类

#### CircleConstraint

```python
class CircleConstraint(TrajectoryConstraint):
    """圆形约束"""

    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        normal: np.ndarray = np.array([0, 0, 1])
    ):
        """
        Args:
            center: [3] 圆心
            radius: 半径
            normal: [3] 法向量
        """
        pass

    def project(self, point: np.ndarray) -> np.ndarray:
        """投影到圆"""
        pass

    def constrain(
        self,
        prior: GaussianDistribution,
        constraint_std_radial: float = 0.01
    ) -> GaussianDistribution:
        """贝叶斯约束"""
        pass
```

### 10.3 配置类

```python
@dataclass
class LSTMConfig:
    """LSTM配置"""
    # 模型
    input_dim: int = 4
    hidden_dim: int = 128
    num_layers: int = 2
    output_dim: int = 3
    dropout: float = 0.1

    # 注意力
    attention_heads: int = 4
    attention_dim: int = 64

    # 训练
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 50
    weight_decay: float = 1e-5

    # 数据
    seq_len: int = 10
    max_cameras: int = 4

    # 不确定性
    min_logvar: float = -10.0
    max_logvar: float = 10.0
```

---

## 附录

### A. 完整示例脚本

见 `test_prediction.py` 和 `colab_prediction_example.py`

### B. 性能基准

**硬件:** NVIDIA T4 GPU, 16GB RAM
- 数据生成: ~2分钟（100帧，3对象，4相机）
- 模型训练: ~5分钟（50 epochs, 32 batch size）
- 预测延迟: ~5ms/sample

### C. 参考文献

1. Kendall & Gal (2017) - Heteroscedastic Uncertainty
2. Vaswani et al. (2017) - Attention Mechanism
3. Hochreiter & Schmidhuber (1997) - LSTM

### D. 更新日志

**v1.0 (2025-11-18)**
- ✅ 完整实现所有功能
- ✅ 修复所有运行时错误
- ✅ 完善文档和示例

---

**文档版本:** v1.0
**最后更新:** 2025-11-18
**维护者:** Smart Monitoring Team

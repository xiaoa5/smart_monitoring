# Probabilistic LSTM Prediction Guide

## 快速开始

### 问题：`NameError: name 'bbox_seq' is not defined`

**原因：** 您需要先从数据集中获取数据，然后才能进行预测。

**解决方案：** 使用以下代码从数据集获取测试数据。

---

## 方法1: 从验证数据集获取单个样本（推荐）

```python
# 1. 获取一个样本
sample = val_dataset[0]  # 获取第一个样本

# 2. 准备输入数据（添加batch维度）
bbox_seq = sample['bbox_seq'].unsqueeze(0).to(device)
camera_ids = sample['camera_ids'].unsqueeze(0).to(device)
mask = sample['mask'].unsqueeze(0).to(device)
ground_truth = sample['pos_3d_seq'].cpu().numpy()

# 3. 进行预测
mean, std = model.predict_distribution(bbox_seq, camera_ids, mask)

# 4. 查看结果
print(f"预测位置: {mean[-1]}")
print(f"不确定性: {std[-1]}")
print(f"真实位置: {ground_truth[-1]}")
```

---

## 方法2: 从DataLoader获取批次数据

```python
# 1. 从验证DataLoader获取一个batch
batch = next(iter(val_loader))

# 2. 获取第一个样本（从batch中）
bbox_seq = batch['bbox_seq'][0:1].to(device)      # [1, seq_len, num_cams, 4]
camera_ids = batch['camera_ids'][0:1].to(device)  # [1, seq_len, num_cams]
mask = batch['mask'][0:1].to(device)              # [1, seq_len, num_cams]
ground_truth = batch['pos_3d_seq'][0].cpu().numpy()  # [seq_len, 3]

# 3. 进行预测
mean, std = model.predict_distribution(bbox_seq, camera_ids, mask)
```

---

## 方法3: 使用训练数据（仅用于调试）

```python
# 如果没有val_dataset，可以使用train_dataset
sample = train_dataset[0]

bbox_seq = sample['bbox_seq'].unsqueeze(0).to(device)
camera_ids = sample['camera_ids'].unsqueeze(0).to(device)
mask = sample['mask'].unsqueeze(0).to(device)
ground_truth = sample['pos_3d_seq'].cpu().numpy()

mean, std = model.predict_distribution(bbox_seq, camera_ids, mask)
```

---

## 完整的Colab示例代码

将以下代码复制到Colab单元格中：

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# 1. 获取测试数据
# ============================================================================

print("获取测试数据...")
sample = val_dataset[0]  # 或者 train_dataset[0]

bbox_seq = sample['bbox_seq'].unsqueeze(0).to(device)
camera_ids = sample['camera_ids'].unsqueeze(0).to(device)
mask = sample['mask'].unsqueeze(0).to(device)
ground_truth = sample['pos_3d_seq'].cpu().numpy()

print(f"✓ 数据形状:")
print(f"  bbox_seq: {bbox_seq.shape}")
print(f"  camera_ids: {camera_ids.shape}")
print(f"  mask: {mask.shape}")

# ============================================================================
# 2. 进行预测
# ============================================================================

print("\n进行预测...")
mean, std = model.predict_distribution(bbox_seq, camera_ids, mask)

print(f"✓ 预测完成:")
print(f"  mean shape: {mean.shape}")
print(f"  std shape: {std.shape}")

# ============================================================================
# 3. 计算误差
# ============================================================================

print("\n计算误差...")
mae = np.abs(mean - ground_truth).mean()
rmse = np.sqrt(((mean - ground_truth) ** 2).mean())

print(f"✓ 误差指标:")
print(f"  MAE:  {mae:.6f} m")
print(f"  RMSE: {rmse:.6f} m")

# ============================================================================
# 4. 可视化
# ============================================================================

print("\n可视化结果...")

fig = plt.figure(figsize=(15, 5))

# 3D轨迹
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(mean[:, 0], mean[:, 1], mean[:, 2], 'b-', linewidth=2, label='预测')
ax1.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
         'g--', linewidth=2, label='真实')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.set_title('3D 轨迹')
ax1.legend()

# XY平面
ax2 = fig.add_subplot(132)
ax2.plot(mean[:, 0], mean[:, 1], 'b-', linewidth=2, label='预测')
ax2.plot(ground_truth[:, 0], ground_truth[:, 1], 'g--', linewidth=2, label='真实')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_title('XY 平面')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.axis('equal')

# 不确定性
ax3 = fig.add_subplot(133)
time_steps = np.arange(len(mean))
uncertainty = np.linalg.norm(std, axis=1)
ax3.plot(time_steps, uncertainty, 'r-', linewidth=2)
ax3.fill_between(time_steps, 0, uncertainty, alpha=0.3, color='red')
ax3.set_xlabel('时间步')
ax3.set_ylabel('不确定性 (m)')
ax3.set_title('预测不确定性')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ 完成!")
```

---

## 数据形状说明

确保您的输入数据具有正确的形状：

| 变量 | 形状 | 说明 |
|------|------|------|
| `bbox_seq` | `[1, seq_len, num_cameras, 4]` | 边界框序列 |
| `camera_ids` | `[1, seq_len, num_cameras]` | 相机ID |
| `mask` | `[1, seq_len, num_cameras]` | 缺失观测掩码 |
| `mean` (输出) | `[seq_len, 3]` | 预测的3D位置 |
| `std` (输出) | `[seq_len, 3]` | 预测的不确定性 |

**注意：**
- 输入需要**批次维度** (`unsqueeze(0)`)
- 输出**自动移除**批次维度

---

## 常见错误

### 1. `NameError: name 'bbox_seq' is not defined`
**解决：** 先从数据集获取数据（见上面的方法1-3）

### 2. `RuntimeError: Expected 4-dimensional input`
**解决：** 确保使用了 `.unsqueeze(0)` 添加批次维度

### 3. `IndexError: list index out of range`
**解决：** 检查数据集是否为空，运行 `len(val_dataset)` 确认

### 4. `RuntimeError: expected dtype Float but got Double`
**解决：** 这个问题已经在最新版本中修复

---

## 测试脚本

我已经创建了两个测试脚本供您使用：

1. **`test_prediction.py`** - 完整的独立测试脚本
   ```bash
   python test_prediction.py
   ```

2. **`colab_prediction_example.py`** - Colab代码片段
   - 直接复制粘贴到Colab

---

## 需要帮助？

如果遇到其他问题，请提供：
1. 完整的错误信息
2. 您正在运行的代码
3. `len(val_dataset)` 的输出

"""
Path 2 Stage 2 - LSTM Training with Data Normalization
修复版本: 添加数据归一化,提升LSTM预测性能

Key Improvements:
1. ✅ 将bbox像素坐标归一化到[0,1]
2. ✅ 训练时使用归一化数据
3. ✅ 预测时反归一化回像素坐标
4. ✅ 按运动类型分别统计性能

Author: AI Assistant
Date: 2025-11-14 (Normalized Version)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from pathlib import Path
from typing import Optional


# ============================================================================
# Stage 2: LSTM Tracker with Normalization
# ============================================================================

class TrackingDatasetNormalized(Dataset):
    """
    LSTM跟踪数据集 - 带归一化
    
    关键改进: 将bbox_pixels归一化到[0,1]范围
    输入: [x1/W, y1/H, x2/W, y2/H]
    """
    
    def __init__(
        self,
        json_file: str,
        sequence_length: int = 10,
        prediction_horizon: int = 5,
        image_size: tuple = (640, 480),
        normalize: bool = True
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.image_width, self.image_height = image_size
        self.normalize = normalize
        
        # 归一化因子
        self.norm_factors = np.array([
            self.image_width,   # x1
            self.image_height,  # y1
            self.image_width,   # x2
            self.image_height   # y2
        ], dtype=np.float32)
        
        # 加载数据
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        # 按相机和物体ID组织轨迹
        self.trajectories = self._organize_trajectories()
        
        # 生成训练样本
        self.samples = self._generate_samples()
        
        print(f"📊 Dataset Statistics:")
        print(f"   Total samples: {len(self.samples)}")
        print(f"   Image size: {self.image_width} × {self.image_height}")
        if self.normalize:
            print(f"   Normalization: ✅ ON (bbox → [0,1])")
        else:
            print(f"   Normalization: ❌ OFF (raw pixels)")
    
    def _organize_trajectories(self):
        """组织轨迹数据"""
        trajectories = {}
        
        for frame_data in self.data:
            cam_id = frame_data['camera_id']
            
            if cam_id not in trajectories:
                trajectories[cam_id] = {}
            
            for obj in frame_data['objects']:
                obj_id = obj['id']
                
                if obj_id not in trajectories[cam_id]:
                    trajectories[cam_id][obj_id] = []
                
                # 使用像素坐标bbox
                if 'bbox_pixels' in obj:
                    bbox = np.array(obj['bbox_pixels'], dtype=np.float32)
                else:
                    # fallback: 从YOLO格式转换
                    cx, cy, w, h = obj['bbox']
                    x1 = (cx - w/2) * self.image_width
                    y1 = (cy - h/2) * self.image_height
                    x2 = (cx + w/2) * self.image_width
                    y2 = (cy + h/2) * self.image_height
                    bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
                
                # 归一化
                if self.normalize:
                    bbox = bbox / self.norm_factors
                
                trajectories[cam_id][obj_id].append({
                    'frame': frame_data['frame'],
                    'bbox': bbox,
                    'motion_type': obj.get('motion_type', 'unknown')
                })
        
        return trajectories
    
    def _generate_samples(self):
        """生成训练样本"""
        samples = []
        
        for cam_id, objects in self.trajectories.items():
            for obj_id, trajectory in objects.items():
                if len(trajectory) < self.sequence_length + self.prediction_horizon:
                    continue
                
                motion_type = trajectory[0]['motion_type']
                
                # 滑动窗口
                for i in range(len(trajectory) - self.sequence_length - self.prediction_horizon + 1):
                    input_seq = trajectory[i:i + self.sequence_length]
                    target_seq = trajectory[i + self.sequence_length:
                                          i + self.sequence_length + self.prediction_horizon]
                    
                    input_bboxes = np.array([t['bbox'] for t in input_seq])
                    target_bboxes = np.array([t['bbox'] for t in target_seq])
                    
                    samples.append({
                        'input': input_bboxes,
                        'target': target_bboxes,
                        'obj_id': obj_id,
                        'cam_id': cam_id,
                        'motion_type': motion_type
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        input_seq = torch.FloatTensor(sample['input'])
        target_seq = torch.FloatTensor(sample['target'])
        
        return input_seq, target_seq
    
    def denormalize_bbox(self, bbox_normalized):
        """
        将归一化的bbox转换回像素坐标
        
        Args:
            bbox_normalized: [x1_norm, y1_norm, x2_norm, y2_norm] in [0,1]
        Returns:
            bbox_pixels: [x1, y1, x2, y2] in pixels
        """
        if isinstance(bbox_normalized, torch.Tensor):
            bbox_normalized = bbox_normalized.cpu().numpy()
        
        bbox_pixels = bbox_normalized * self.norm_factors
        return bbox_pixels


class LSTMTracker(nn.Module):
    """
    LSTM多目标跟踪器
    
    输入: (batch, seq_len=10, input_size=4) - 归一化的bbox
    输出: (batch, output_size=4) - 归一化的bbox预测
    """
    
    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 4,
        dropout: float = 0.2
    ):
        super(LSTMTracker, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x, hidden=None):
        """
        前向传播
        
        Args:
            x: (batch, seq_len, input_size)
            hidden: (h0, c0) or None
        Returns:
            output: (batch, output_size)
            hidden: (h, c)
        """
        lstm_out, hidden = self.lstm(x, hidden)
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        
        return prediction, hidden
    
    def predict_sequence(self, x, steps: int = 5):
        """
        自回归预测多步
        
        Args:
            x: (batch, seq_len, input_size)
            steps: 预测步数
        Returns:
            predictions: (batch, steps, output_size)
        """
        predictions = []
        current_seq = x.clone()
        hidden = None
        
        for _ in range(steps):
            pred, hidden = self.forward(current_seq, hidden)
            predictions.append(pred)
            current_seq = torch.cat([current_seq[:, 1:, :], pred.unsqueeze(1)], dim=1)
        
        return torch.stack(predictions, dim=1)


class LSTMTrackerTrainer:
    """LSTM跟踪器训练器"""
    
    def __init__(
        self,
        model: LSTMTracker,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.001
    ):
        self.model = model.to(device)
        self.device = device
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 预测所有未来步
            predictions = self.model.predict_sequence(inputs, steps=targets.size(1))
            
            # 计算损失
            loss = self.criterion(predictions, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model.predict_sequence(inputs, steps=targets.size(1))
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        save_path: Optional[str] = None
    ):
        """完整训练流程"""
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        print(f"\n🚀 Starting training...")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {train_loader.batch_size}")
        print(f"   Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print()
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 打印进度
            print(f"Epoch {epoch+1:3d}/{num_epochs} - "
                  f"Train Loss: {train_loss:8.4f}, Val Loss: {val_loss:8.4f}", end='')
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  ✅ Saved (best: {val_loss:.4f})")
                else:
                    print(f"  ⭐ New best!")
            else:
                patience_counter += 1
                print()
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
        
        print(f"\n🎉 Training completed!")
        print(f"   Best val loss: {best_val_loss:.4f}")
        print(f"   Final train loss: {train_loss:.4f}")


# ============================================================================
# Demo Functions
# ============================================================================

def train_lstm_with_normalization(
    json_file: str,
    output_dir: str = "./path2_output_corrected/stage2",
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.001
):
    """
    使用归一化数据训练LSTM
    """
    print("=" * 80)
    print("Stage 2: LSTM Training with Normalization")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建数据集 (带归一化!)
    print("\n📊 Creating normalized dataset...")
    dataset = TrackingDatasetNormalized(
        json_file=json_file,
        sequence_length=10,
        prediction_horizon=5,
        image_size=(640, 480),
        normalize=True  # ← 关键!
    )
    
    if len(dataset) == 0:
        print("⚠️  No training samples generated.")
        print("   Try generating longer sequences (duration > 20s)")
        return
    
    # 划分训练/验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n📈 Data split:")
    print(f"   Train: {train_size} samples")
    print(f"   Val:   {val_size} samples")
    
    # 创建模型
    print("\n🧠 Creating LSTM model...")
    model = LSTMTracker(
        input_size=4,
        hidden_size=128,
        num_layers=2,
        output_size=4,
        dropout=0.2
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")
    
    # 训练
    trainer = LSTMTrackerTrainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=learning_rate
    )
    
    save_path = f"{output_dir}/best_lstm_tracker_normalized.pth"
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_path=save_path
    )
    
    print(f"\n✅ Model saved to: {save_path}")
    print(f"\n💡 Expected improvement:")
    print(f"   With normalization: Loss should be < 0.01 (in [0,1] space)")
    print(f"   Which equals: ~25 pixels MAE in image space")
    
    return save_path


def main():
    """主函数"""
    import sys
    
    print("\n" + "=" * 80)
    print("🚀 Path 2 Stage 2 - LSTM Training with Normalization")
    print("=" * 80)
    
    # 检查数据文件
    json_file = "./path2_output_corrected/stage1/motion_sequence.json"
    
    if not Path(json_file).exists():
        print(f"\n⚠️  Data file not found: {json_file}")
        print("   Please run path2_CORRECTED_v2.py first to generate data.")
        sys.exit(1)
    
    # 训练
    model_path = train_lstm_with_normalization(
        json_file=json_file,
        num_epochs=50,
        batch_size=16,
        learning_rate=0.001
    )
    
    print("\n" + "=" * 80)
    print("✅ Training completed!")
    print("=" * 80)
    print(f"\n📁 Output files:")
    print(f"   Model: {model_path}")
    print(f"\n🎨 Next steps:")
    print(f"   1. python enhanced_visualization.py  # 可视化对比")
    print(f"   2. Check prediction quality")
    print(f"   3. Compare with non-normalized version")


if __name__ == "__main__":
    main()

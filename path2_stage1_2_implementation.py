"""
Path 2 Implementation: Stage 1 & 2
Motion Sequence Generator + LSTM-Based Multi-Object Tracker

Author: AI Assistant
Date: 2025-11-14
Status: Ready for Testing

Dependencies:
    pip install pybullet numpy torch opencv-python matplotlib pyyaml --break-system-packages
"""

import pybullet as p
import pybullet_data
import numpy as np
import cv2
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import math
from dataclasses import dataclass, asdict
from enum import Enum

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Stage 1: Motion Sequence Generator
# ============================================================================

class MotionType(Enum):
    """支持的运动类型"""
    LINEAR = "linear"
    CIRCULAR = "circular"
    RANDOM_WALK = "random_walk"
    STATIONARY = "stationary"


@dataclass
class ObjectState:
    """物体状态"""
    id: int
    pos_3d: List[float]  # [x, y, z] in world coordinates
    bbox: List[float]    # [x1, y1, x2, y2] in image coordinates
    occlusion: float     # 0.0 (visible) to 1.0 (fully occluded)
    velocity: List[float]  # [vx, vy, vz]
    motion_type: str


@dataclass
class FrameData:
    """单帧数据"""
    frame: int
    timestamp: float
    camera_id: int
    objects: List[ObjectState]


class MotionSequenceGenerator:
    """
    阶段1: 运动序列生成器
    生成可控的多目标连续轨迹数据
    """
    
    def __init__(
        self,
        scene_size: Tuple[float, float] = (10.0, 10.0),
        num_cameras: int = 4,
        fps: int = 30,
        output_dir: str = "./motion_sequences"
    ):
        self.scene_size = scene_size
        self.num_cameras = num_cameras
        self.fps = fps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # PyBullet初始化
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        
        # 场景设置
        self.plane_id = p.loadURDF("plane.urdf")
        self.object_ids = []
        self.object_motions = {}
        
        # 相机设置
        self.cameras = self._setup_cameras()
        
    def _setup_cameras(self) -> List[Dict]:
        """设置多相机配置"""
        cameras = []
        width, height = 640, 480
        fov = 60
        aspect = width / height
        near, far = 0.1, 100
        
        # 四周相机布局
        positions = [
            [self.scene_size[0]/2, -2, 3],   # South
            [self.scene_size[0]/2, self.scene_size[1]+2, 3],  # North
            [-2, self.scene_size[1]/2, 3],   # West
            [self.scene_size[0]+2, self.scene_size[1]/2, 3]   # East
        ]
        
        targets = [[self.scene_size[0]/2, self.scene_size[1]/2, 0]] * 4
        
        for i in range(self.num_cameras):
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=positions[i],
                cameraTargetPosition=targets[i],
                cameraUpVector=[0, 0, 1]
            )
            
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=fov, aspect=aspect, nearVal=near, farVal=far
            )
            
            cameras.append({
                'id': i,
                'position': positions[i],
                'target': targets[i],
                'view_matrix': view_matrix,
                'proj_matrix': proj_matrix,
                'width': width,
                'height': height,
                'fov': fov
            })
            
        return cameras
    
    def add_object(
        self,
        obj_id: int,
        start_pos: List[float],
        motion_type: MotionType,
        **motion_params
    ):
        """添加运动物体"""
        # 创建简单的立方体物体
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3])
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.3, 0.3, 0.3],
            rgbaColor=[np.random.rand(), np.random.rand(), np.random.rand(), 1]
        )
        
        body_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=start_pos
        )
        
        self.object_ids.append(body_id)
        
        # 保存运动参数
        self.object_motions[body_id] = {
            'id': obj_id,
            'type': motion_type,
            'start_pos': start_pos,
            'params': motion_params,
            'time': 0.0
        }
        
        return body_id
    
    def _update_object_motion(self, body_id: int, dt: float):
        """更新物体运动"""
        motion = self.object_motions[body_id]
        motion['time'] += dt
        
        t = motion['time']
        motion_type = motion['type']
        params = motion['params']
        start_pos = motion['start_pos']
        
        if motion_type == MotionType.LINEAR:
            # 直线运动
            velocity = params.get('velocity', [1.0, 0.0, 0.0])
            new_pos = [
                start_pos[0] + velocity[0] * t,
                start_pos[1] + velocity[1] * t,
                start_pos[2] + velocity[2] * t
            ]
            
        elif motion_type == MotionType.CIRCULAR:
            # 圆周运动
            center = params.get('center', [5.0, 5.0, 0.5])
            radius = params.get('radius', 2.0)
            angular_velocity = params.get('angular_velocity', 1.0)
            
            angle = angular_velocity * t
            new_pos = [
                center[0] + radius * np.cos(angle),
                center[1] + radius * np.sin(angle),
                center[2]
            ]
            
        elif motion_type == MotionType.RANDOM_WALK:
            # 随机游走
            step_size = params.get('step_size', 0.1)
            current_pos, _ = p.getBasePositionAndOrientation(body_id)
            
            # 添加随机扰动
            noise = np.random.randn(3) * step_size
            new_pos = [
                np.clip(current_pos[0] + noise[0], 0, self.scene_size[0]),
                np.clip(current_pos[1] + noise[1], 0, self.scene_size[1]),
                current_pos[2]
            ]
            
        else:  # STATIONARY
            new_pos = start_pos
        
        p.resetBasePositionAndOrientation(
            body_id,
            new_pos,
            [0, 0, 0, 1]
        )
        
        return new_pos
    
    def _compute_occlusion(self, obj_pos: List[float], camera: Dict) -> float:
        """计算遮挡程度(简化版)"""
        # 使用深度图计算遮挡
        # 这里简化实现,实际应该用raycast
        return np.random.uniform(0, 0.3)  # 简化: 随机遮挡
    
    def _project_to_image(
        self,
        pos_3d: List[float],
        camera: Dict
    ) -> Optional[List[float]]:
        """3D点投影到图像坐标"""
        # 简化的投影,实际应该用完整的相机矩阵
        cam_pos = camera['position']
        width, height = camera['width'], camera['height']
        
        # 计算相对位置
        dx = pos_3d[0] - cam_pos[0]
        dy = pos_3d[1] - cam_pos[1]
        dz = pos_3d[2] - cam_pos[2]
        
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        if dist < 0.1:
            return None
        
        # 简化的透视投影
        fov = camera['fov']
        scale = height / (2 * np.tan(np.radians(fov) / 2))
        
        x = width/2 + (dx / dist) * scale
        y = height/2 - (dz / dist) * scale
        
        # 边界框(假设固定大小)
        box_size = 50 / dist  # 近大远小
        
        bbox = [
            max(0, x - box_size),
            max(0, y - box_size),
            min(width, x + box_size),
            min(height, y + box_size)
        ]
        
        # 检查是否在视野内
        if bbox[2] > 0 and bbox[0] < width and bbox[3] > 0 and bbox[1] < height:
            return bbox
        return None
    
    def generate_sequence(
        self,
        duration: float = 10.0,
        save_video: bool = True,
        save_json: bool = True
    ) -> List[FrameData]:
        """生成运动序列"""
        dt = 1.0 / self.fps
        num_frames = int(duration * self.fps)
        
        all_frame_data = []
        
        for frame_idx in range(num_frames):
            t = frame_idx * dt
            
            # 更新物体运动
            for body_id in self.object_ids:
                self._update_object_motion(body_id, dt)
            
            p.stepSimulation()
            
            # 对每个相机生成数据
            for camera in self.cameras:
                objects_in_frame = []
                
                for body_id in self.object_ids:
                    pos_3d, _ = p.getBasePositionAndOrientation(body_id)
                    
                    # 投影到图像
                    bbox = self._project_to_image(list(pos_3d), camera)
                    if bbox is None:
                        continue
                    
                    # 计算遮挡
                    occlusion = self._compute_occlusion(pos_3d, camera)
                    
                    # 计算速度(简化)
                    motion = self.object_motions[body_id]
                    if motion['type'] == MotionType.LINEAR:
                        velocity = motion['params'].get('velocity', [0, 0, 0])
                    else:
                        velocity = [0, 0, 0]  # 简化
                    
                    obj_state = ObjectState(
                        id=motion['id'],
                        pos_3d=list(pos_3d),
                        bbox=bbox,
                        occlusion=occlusion,
                        velocity=velocity,
                        motion_type=motion['type'].value
                    )
                    
                    objects_in_frame.append(obj_state)
                
                frame_data = FrameData(
                    frame=frame_idx,
                    timestamp=t,
                    camera_id=camera['id'],
                    objects=objects_in_frame
                )
                
                all_frame_data.append(frame_data)
        
        # 保存数据
        if save_json:
            self._save_json(all_frame_data)
        
        if save_video:
            self._save_video(all_frame_data)
        
        return all_frame_data
    
    def _save_json(self, frame_data: List[FrameData]):
        """保存为JSON格式"""
        output_file = self.output_dir / "motion_sequence.json"
        
        data_list = []
        for fd in frame_data:
            frame_dict = {
                'frame': fd.frame,
                'timestamp': fd.timestamp,
                'camera_id': fd.camera_id,
                'objects': [asdict(obj) for obj in fd.objects]
            }
            data_list.append(frame_dict)
        
        with open(output_file, 'w') as f:
            json.dump(data_list, f, indent=2)
        
        print(f"✅ Saved motion sequence to {output_file}")
    
    def _save_video(self, frame_data: List[FrameData]):
        """保存可视化视频(简化版)"""
        print("📹 Video generation skipped in this version (requires rendering)")
    
    def cleanup(self):
        """清理资源"""
        p.disconnect(self.client)


# ============================================================================
# Stage 2: LSTM-Based Multi-Object Tracker
# ============================================================================

class TrackingDataset(Dataset):
    """LSTM跟踪数据集"""
    
    def __init__(
        self,
        json_file: str,
        sequence_length: int = 10,
        prediction_horizon: int = 5
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # 加载数据
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        # 按相机和物体ID组织轨迹
        self.trajectories = self._organize_trajectories()
        
        # 生成训练样本
        self.samples = self._generate_samples()
    
    def _organize_trajectories(self) -> Dict:
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
                
                # 提取bbox和位置信息
                trajectories[cam_id][obj_id].append({
                    'frame': frame_data['frame'],
                    'bbox': obj['bbox'],
                    'pos_3d': obj['pos_3d'],
                    'velocity': obj['velocity']
                })
        
        return trajectories
    
    def _generate_samples(self) -> List[Dict]:
        """生成训练样本"""
        samples = []
        
        for cam_id, objects in self.trajectories.items():
            for obj_id, trajectory in objects.items():
                # 确保轨迹足够长
                if len(trajectory) < self.sequence_length + self.prediction_horizon:
                    continue
                
                # 滑动窗口
                for i in range(len(trajectory) - self.sequence_length - self.prediction_horizon + 1):
                    input_seq = trajectory[i:i + self.sequence_length]
                    target_seq = trajectory[i + self.sequence_length:
                                          i + self.sequence_length + self.prediction_horizon]
                    
                    # 提取bbox序列 [x1, y1, x2, y2]
                    input_bboxes = np.array([t['bbox'] for t in input_seq])
                    target_bboxes = np.array([t['bbox'] for t in target_seq])
                    
                    samples.append({
                        'input': input_bboxes,
                        'target': target_bboxes,
                        'obj_id': obj_id,
                        'cam_id': cam_id
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        input_seq = torch.FloatTensor(sample['input'])
        target_seq = torch.FloatTensor(sample['target'])
        
        return input_seq, target_seq


class LSTMTracker(nn.Module):
    """
    阶段2: LSTM多目标跟踪器
    预测未来的边界框或世界坐标位置
    """
    
    def __init__(
        self,
        input_size: int = 4,      # bbox: [x1, y1, x2, y2]
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
        Args:
            x: (batch, seq_len, input_size)
            hidden: (h0, c0) or None
        Returns:
            output: (batch, output_size)
            hidden: (h, c)
        """
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)
        
        # 取最后一个时间步
        last_output = lstm_out[:, -1, :]
        
        # 全连接预测
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
            # 预测下一步
            pred, hidden = self.forward(current_seq, hidden)
            predictions.append(pred)
            
            # 更新序列(滑动窗口)
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
            
            # 前向传播
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
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # 保存最佳模型
            if save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"  ✅ Saved best model (val_loss: {val_loss:.6f})")
        
        print(f"\n🎉 Training completed! Best val loss: {best_val_loss:.6f}")


# ============================================================================
# Integration & Demo
# ============================================================================

def run_stage1_demo():
    """阶段1演示: 生成运动序列"""
    print("=" * 60)
    print("Stage 1: Motion Sequence Generator Demo")
    print("=" * 60)
    
    generator = MotionSequenceGenerator(
        scene_size=(10.0, 10.0),
        num_cameras=4,
        fps=30,
        output_dir="./path2_output/stage1"
    )
    
    # 添加不同运动模式的物体
    print("\n📦 Adding objects with different motion patterns...")
    
    # 1. 直线运动
    generator.add_object(
        obj_id=1,
        start_pos=[1.0, 1.0, 0.5],
        motion_type=MotionType.LINEAR,
        velocity=[0.5, 0.3, 0.0]
    )
    
    # 2. 圆周运动
    generator.add_object(
        obj_id=2,
        start_pos=[7.0, 5.0, 0.5],
        motion_type=MotionType.CIRCULAR,
        center=[5.0, 5.0, 0.5],
        radius=2.0,
        angular_velocity=0.5
    )
    
    # 3. 随机游走
    generator.add_object(
        obj_id=3,
        start_pos=[3.0, 7.0, 0.5],
        motion_type=MotionType.RANDOM_WALK,
        step_size=0.05
    )
    
    # 生成序列
    print("\n🎬 Generating motion sequences...")
    frame_data = generator.generate_sequence(
        duration=10.0,
        save_video=False,
        save_json=True
    )
    
    print(f"\n✅ Generated {len(frame_data)} frames of data")
    print(f"   Saved to: ./path2_output/stage1/motion_sequence.json")
    
    generator.cleanup()
    
    return "./path2_output/stage1/motion_sequence.json"


def run_stage2_demo(json_file: str):
    """阶段2演示: 训练LSTM跟踪器"""
    print("\n" + "=" * 60)
    print("Stage 2: LSTM-Based Multi-Object Tracker Demo")
    print("=" * 60)
    
    # 创建数据集
    print("\n📊 Creating dataset...")
    dataset = TrackingDataset(
        json_file=json_file,
        sequence_length=10,
        prediction_horizon=5
    )
    
    print(f"   Total samples: {len(dataset)}")
    
    if len(dataset) == 0:
        print("⚠️  No training samples generated. Need longer sequences.")
        return
    
    # 划分训练/验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"   Train samples: {train_size}, Val samples: {val_size}")
    
    # 创建模型
    print("\n🧠 Creating LSTM Tracker model...")
    model = LSTMTracker(
        input_size=4,
        hidden_size=128,
        num_layers=2,
        output_size=4,
        dropout=0.2
    )
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    print("\n🚀 Starting training...")
    trainer = LSTMTrackerTrainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=0.001
    )
    
    os.makedirs("./path2_output/stage2", exist_ok=True)
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=30,
        save_path="./path2_output/stage2/best_lstm_tracker.pth"
    )
    
    print("\n✅ Stage 2 completed!")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🚀 Path 2 Implementation: Stage 1 & 2")
    print("=" * 60)
    
    # 运行阶段1
    json_file = run_stage1_demo()
    
    # 运行阶段2
    run_stage2_demo(json_file)
    
    print("\n" + "=" * 60)
    print("🎉 All stages completed successfully!")
    print("=" * 60)
    print("\n📁 Output files:")
    print("   Stage 1: ./path2_output/stage1/motion_sequence.json")
    print("   Stage 2: ./path2_output/stage2/best_lstm_tracker.pth")
    print("\n💡 Next steps:")
    print("   1. Visualize the generated motion sequences")
    print("   2. Test the LSTM tracker on new sequences")
    print("   3. Integrate with YOLO detection pipeline")
    print("   4. Add ReID for identity consistency (Stage 3)")


if __name__ == "__main__":
    main()

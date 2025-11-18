"""
Path 2 Visualization & Testing Tools
可视化运动序列和LSTM预测结果
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import torch
from pathlib import Path


def visualize_motion_sequence(json_file: str, camera_id: int = 0, save_gif: bool = False):
    """
    可视化运动序列
    
    Args:
        json_file: 运动序列JSON文件路径
        camera_id: 要可视化的相机ID
        save_gif: 是否保存为GIF动画
    """
    # 加载数据
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 过滤指定相机的数据
    camera_data = [d for d in data if d['camera_id'] == camera_id]
    
    if not camera_data:
        print(f"⚠️  No data found for camera {camera_id}")
        return
    
    print(f"📊 Visualizing {len(camera_data)} frames from camera {camera_id}")
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图: 2D轨迹
    ax1.set_title(f'Camera {camera_id} - 2D Trajectories')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.set_xlim(0, 640)
    ax1.set_ylim(480, 0)  # 图像坐标系Y轴向下
    ax1.grid(True, alpha=0.3)
    
    # 右图: 3D轨迹
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('World Space - 3D Trajectories')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    
    # 收集所有物体的轨迹
    trajectories_2d = {}
    trajectories_3d = {}
    
    for frame_data in camera_data:
        for obj in frame_data['objects']:
            obj_id = obj['id']
            
            if obj_id not in trajectories_2d:
                trajectories_2d[obj_id] = {'x': [], 'y': []}
                trajectories_3d[obj_id] = {'x': [], 'y': [], 'z': []}
            
            # 2D bbox中心
            bbox = obj['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            trajectories_2d[obj_id]['x'].append(cx)
            trajectories_2d[obj_id]['y'].append(cy)
            
            # 3D位置
            pos_3d = obj['pos_3d']
            trajectories_3d[obj_id]['x'].append(pos_3d[0])
            trajectories_3d[obj_id]['y'].append(pos_3d[1])
            trajectories_3d[obj_id]['z'].append(pos_3d[2])
    
    # 绘制轨迹
    colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories_2d)))
    
    for (obj_id, traj_2d), (_, traj_3d), color in zip(
        trajectories_2d.items(), 
        trajectories_3d.items(), 
        colors
    ):
        # 2D轨迹
        ax1.plot(traj_2d['x'], traj_2d['y'], 
                label=f'Object {obj_id}', 
                color=color, 
                linewidth=2, 
                alpha=0.7)
        ax1.scatter(traj_2d['x'][0], traj_2d['y'][0], 
                   color=color, 
                   s=100, 
                   marker='o', 
                   label=f'Start {obj_id}')
        ax1.scatter(traj_2d['x'][-1], traj_2d['y'][-1], 
                   color=color, 
                   s=100, 
                   marker='X', 
                   label=f'End {obj_id}')
        
        # 3D轨迹
        ax2.plot(traj_3d['x'], traj_3d['y'], traj_3d['z'], 
                color=color, 
                linewidth=2, 
                alpha=0.7)
        ax2.scatter(traj_3d['x'][0], traj_3d['y'][0], traj_3d['z'][0], 
                   color=color, 
                   s=100, 
                   marker='o')
        ax2.scatter(traj_3d['x'][-1], traj_3d['y'][-1], traj_3d['z'][-1], 
                   color=color, 
                   s=100, 
                   marker='X')
    
    ax1.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save_gif:
        output_path = Path(json_file).parent / f"camera_{camera_id}_trajectories.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to {output_path}")
    
    plt.show()


def analyze_lstm_predictions(
    model_path: str,
    json_file: str,
    num_samples: int = 5
):
    """
    分析LSTM预测结果
    
    Args:
        model_path: 训练好的模型路径
        json_file: 测试数据JSON文件
        num_samples: 可视化样本数量
    """
    from path2_stage1_2_implementation import LSTMTracker, TrackingDataset
    
    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMTracker(
        input_size=4,
        hidden_size=128,
        num_layers=2,
        output_size=4
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✅ Loaded model from {model_path}")
    
    # 创建数据集
    dataset = TrackingDataset(
        json_file=json_file,
        sequence_length=10,
        prediction_horizon=5
    )
    
    print(f"📊 Dataset size: {len(dataset)} samples")
    
    if len(dataset) == 0:
        print("⚠️  No samples in dataset")
        return
    
    # 随机选择样本
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            input_seq, target_seq = dataset[idx]
            
            # 预测
            input_tensor = input_seq.unsqueeze(0).to(device)
            predictions = model.predict_sequence(input_tensor, steps=target_seq.size(0))
            predictions = predictions.squeeze(0).cpu().numpy()
            
            # 转换为numpy
            input_np = input_seq.numpy()
            target_np = target_seq.numpy()
            
            # 可视化
            ax = axes[i]
            
            # 输入序列
            time_input = np.arange(len(input_np))
            ax.plot(time_input, input_np[:, 0], 'b-', label='Input X1', alpha=0.7)
            ax.plot(time_input, input_np[:, 2], 'g-', label='Input X2', alpha=0.7)
            
            # 目标序列
            time_target = np.arange(len(input_np), len(input_np) + len(target_np))
            ax.plot(time_target, target_np[:, 0], 'b--', label='Target X1', linewidth=2)
            ax.plot(time_target, target_np[:, 2], 'g--', label='Target X2', linewidth=2)
            
            # 预测序列
            ax.plot(time_target, predictions[:, 0], 'r-', label='Pred X1', linewidth=2)
            ax.plot(time_target, predictions[:, 2], 'orange', label='Pred X2', linewidth=2)
            
            # 计算误差
            mse = np.mean((predictions - target_np) ** 2)
            
            ax.set_title(f'Sample {i+1} - MSE: {mse:.4f}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Pixel Coordinate')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(model_path).parent / "prediction_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved analysis to {output_path}")
    
    plt.show()


def compute_tracking_metrics(
    model_path: str,
    json_file: str
):
    """
    计算跟踪性能指标
    
    Args:
        model_path: 模型路径
        json_file: 测试数据路径
    """
    from path2_stage1_2_implementation import LSTMTracker, TrackingDataset
    
    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMTracker(
        input_size=4,
        hidden_size=128,
        num_layers=2,
        output_size=4
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 创建数据集
    dataset = TrackingDataset(
        json_file=json_file,
        sequence_length=10,
        prediction_horizon=5
    )
    
    if len(dataset) == 0:
        print("⚠️  No samples for evaluation")
        return
    
    print("🔍 Computing tracking metrics...")
    
    # 计算指标
    all_errors = []
    all_ious = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            input_seq, target_seq = dataset[i]
            
            # 预测
            input_tensor = input_seq.unsqueeze(0).to(device)
            predictions = model.predict_sequence(input_tensor, steps=target_seq.size(0))
            predictions = predictions.squeeze(0).cpu().numpy()
            target_np = target_seq.numpy()
            
            # L2误差
            error = np.sqrt(np.sum((predictions - target_np) ** 2, axis=1))
            all_errors.extend(error)
            
            # IoU (简化计算)
            for pred_box, target_box in zip(predictions, target_np):
                iou = compute_iou(pred_box, target_box)
                all_ious.append(iou)
    
    # 打印统计
    print("\n" + "=" * 50)
    print("📊 Tracking Performance Metrics")
    print("=" * 50)
    print(f"Total Predictions: {len(all_errors)}")
    print(f"\nL2 Error (pixels):")
    print(f"  Mean:   {np.mean(all_errors):.2f}")
    print(f"  Median: {np.median(all_errors):.2f}")
    print(f"  Std:    {np.std(all_errors):.2f}")
    print(f"  Min:    {np.min(all_errors):.2f}")
    print(f"  Max:    {np.max(all_errors):.2f}")
    print(f"\nIoU:")
    print(f"  Mean:   {np.mean(all_ious):.3f}")
    print(f"  Median: {np.median(all_ious):.3f}")
    print(f"  >0.5:   {np.sum(np.array(all_ious) > 0.5) / len(all_ious) * 100:.1f}%")
    print("=" * 50)


def compute_iou(box1, box2):
    """计算IoU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 计算并集
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def main():
    """演示所有可视化功能"""
    print("🎨 Path 2 Visualization Tools\n")
    
    json_file = "./path2_output/stage1/motion_sequence.json"
    model_file = "./path2_output/stage2/best_lstm_tracker.pth"
    
    if not Path(json_file).exists():
        print(f"⚠️  Data file not found: {json_file}")
        print("   Please run path2_stage1_2_implementation.py first")
        return
    
    # 1. 可视化运动序列
    print("1️⃣  Visualizing motion sequences...")
    visualize_motion_sequence(json_file, camera_id=0, save_gif=True)
    
    if not Path(model_file).exists():
        print(f"\n⚠️  Model file not found: {model_file}")
        print("   Skipping LSTM analysis")
        return
    
    # 2. 分析LSTM预测
    print("\n2️⃣  Analyzing LSTM predictions...")
    analyze_lstm_predictions(model_file, json_file, num_samples=3)
    
    # 3. 计算性能指标
    print("\n3️⃣  Computing tracking metrics...")
    compute_tracking_metrics(model_file, json_file)
    
    print("\n✅ All visualizations completed!")


if __name__ == "__main__":
    main()

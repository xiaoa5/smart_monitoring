"""
Enhanced Visualization Tool
显示真实轨迹 vs LSTM预测的完整对比
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from pathlib import Path


def visualize_prediction_vs_ground_truth(
    model_path: str,
    json_file: str,
    num_samples: int = 3,
    image_size: tuple = (640, 480)
):
    """
    可视化LSTM预测 vs 真实轨迹
    
    显示:
    1. 完整的真实轨迹 (蓝色)
    2. 输入序列 (绿色)
    3. LSTM预测 (红色)
    4. 真实未来轨迹 (橙色虚线)
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
    
    # 加载完整轨迹数据
    with open(json_file, 'r') as f:
        all_data = json.load(f)
    
    # 组织轨迹
    trajectories = {}
    for frame_data in all_data:
        cam_id = frame_data['camera_id']
        if cam_id not in trajectories:
            trajectories[cam_id] = {}
        
        for obj in frame_data['objects']:
            obj_id = obj['id']
            if obj_id not in trajectories[cam_id]:
                trajectories[cam_id][obj_id] = []
            
            # 使用像素坐标
            if 'bbox_pixels' in obj:
                bbox = obj['bbox_pixels']
            else:
                cx, cy, w, h = obj['bbox']
                x1 = (cx - w/2) * image_size[0]
                y1 = (cy - h/2) * image_size[1]
                x2 = (cx + w/2) * image_size[0]
                y2 = (cy + h/2) * image_size[1]
                bbox = [x1, y1, x2, y2]
            
            trajectories[cam_id][obj_id].append({
                'frame': frame_data['frame'],
                'bbox': bbox,
                'motion_type': obj.get('motion_type', 'unknown')
            })
    
    # 随机选择样本
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # 创建大图
    fig = plt.figure(figsize=(20, 6*num_samples))
    
    with torch.no_grad():
        for plot_idx, idx in enumerate(indices):
            input_seq, target_seq = dataset[idx]
            sample_info = dataset.samples[idx]
            
            obj_id = sample_info['obj_id']
            cam_id = sample_info['cam_id']
            
            # 预测
            input_tensor = input_seq.unsqueeze(0).to(device)
            predictions = model.predict_sequence(input_tensor, steps=target_seq.size(0))
            predictions = predictions.squeeze(0).cpu().numpy()
            
            # 转换为numpy
            input_np = input_seq.numpy()
            target_np = target_seq.numpy()
            
            # 获取完整轨迹
            full_trajectory = trajectories[cam_id][obj_id]
            full_bboxes = np.array([t['bbox'] for t in full_trajectory])
            motion_type = full_trajectory[0]['motion_type']
            
            # 计算误差
            mse = np.mean((predictions - target_np) ** 2)
            mae = np.mean(np.abs(predictions - target_np))
            
            # ============= 创建子图 =============
            
            # 子图1: 2D轨迹可视化 (鸟瞰图)
            ax1 = plt.subplot(num_samples, 3, plot_idx*3 + 1)
            
            # 完整真实轨迹 (淡蓝色)
            centers_x = (full_bboxes[:, 0] + full_bboxes[:, 2]) / 2
            centers_y = (full_bboxes[:, 1] + full_bboxes[:, 3]) / 2
            ax1.plot(centers_x, centers_y, 'lightblue', alpha=0.3, linewidth=3, 
                    label='Full Ground Truth Trajectory')
            
            # 输入序列 (绿色)
            input_centers_x = (input_np[:, 0] + input_np[:, 2]) / 2
            input_centers_y = (input_np[:, 1] + input_np[:, 3]) / 2
            ax1.plot(input_centers_x, input_centers_y, 'g-o', linewidth=2, 
                    markersize=6, label='Input Sequence (10 frames)')
            
            # 真实未来轨迹 (橙色虚线)
            target_centers_x = (target_np[:, 0] + target_np[:, 2]) / 2
            target_centers_y = (target_np[:, 1] + target_np[:, 3]) / 2
            ax1.plot(target_centers_x, target_centers_y, 'orange', linestyle='--', 
                    linewidth=2.5, marker='s', markersize=7, label='Ground Truth Future (5 frames)')
            
            # LSTM预测 (红色)
            pred_centers_x = (predictions[:, 0] + predictions[:, 2]) / 2
            pred_centers_y = (predictions[:, 1] + predictions[:, 3]) / 2
            ax1.plot(pred_centers_x, pred_centers_y, 'r-^', linewidth=2.5, 
                    markersize=7, label='LSTM Prediction (5 frames)')
            
            # 绘制bbox (最后一帧)
            # 真实
            last_target = target_np[-1]
            rect_target = Rectangle(
                (last_target[0], last_target[1]),
                last_target[2] - last_target[0],
                last_target[3] - last_target[1],
                linewidth=2, edgecolor='orange', facecolor='none', linestyle='--'
            )
            ax1.add_patch(rect_target)
            
            # 预测
            last_pred = predictions[-1]
            rect_pred = Rectangle(
                (last_pred[0], last_pred[1]),
                last_pred[2] - last_pred[0],
                last_pred[3] - last_pred[1],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax1.add_patch(rect_pred)
            
            ax1.set_xlim(0, image_size[0])
            ax1.set_ylim(image_size[1], 0)  # 图像坐标系
            ax1.set_xlabel('X (pixels)', fontsize=12)
            ax1.set_ylabel('Y (pixels)', fontsize=12)
            ax1.set_title(f'Sample {plot_idx+1} - Camera {cam_id}, Object {obj_id} ({motion_type})\n'
                         f'MSE: {mse:.2f}, MAE: {mae:.2f}', fontsize=12, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
            
            # 子图2: X坐标时序图
            ax2 = plt.subplot(num_samples, 3, plot_idx*3 + 2)
            
            seq_len = len(input_np)
            total_len = seq_len + len(target_np)
            time_input = np.arange(seq_len)
            time_future = np.arange(seq_len, total_len)
            
            # X1坐标
            ax2.plot(time_input, input_np[:, 0], 'g-', linewidth=2, label='Input X1')
            ax2.plot(time_future, target_np[:, 0], 'orange', linestyle='--', 
                    linewidth=2.5, marker='s', markersize=6, label='GT X1')
            ax2.plot(time_future, predictions[:, 0], 'r-', linewidth=2.5, 
                    marker='^', markersize=6, label='Pred X1')
            
            ax2.axvline(x=seq_len-0.5, color='gray', linestyle=':', linewidth=1.5, 
                       label='Prediction Start')
            ax2.set_xlabel('Time Step', fontsize=11)
            ax2.set_ylabel('X1 Coordinate (pixels)', fontsize=11)
            ax2.set_title('X1 (Left Edge) Prediction', fontsize=11)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            
            # 子图3: Y坐标时序图
            ax3 = plt.subplot(num_samples, 3, plot_idx*3 + 3)
            
            # Y1坐标
            ax3.plot(time_input, input_np[:, 1], 'g-', linewidth=2, label='Input Y1')
            ax3.plot(time_future, target_np[:, 1], 'orange', linestyle='--', 
                    linewidth=2.5, marker='s', markersize=6, label='GT Y1')
            ax3.plot(time_future, predictions[:, 1], 'r-', linewidth=2.5, 
                    marker='^', markersize=6, label='Pred Y1')
            
            ax3.axvline(x=seq_len-0.5, color='gray', linestyle=':', linewidth=1.5, 
                       label='Prediction Start')
            ax3.set_xlabel('Time Step', fontsize=11)
            ax3.set_ylabel('Y1 Coordinate (pixels)', fontsize=11)
            ax3.set_title('Y1 (Top Edge) Prediction', fontsize=11)
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(model_path).parent / "prediction_vs_ground_truth.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved comparison to {output_path}")
    
    plt.show()


def compute_detailed_metrics(
    model_path: str,
    json_file: str
):
    """
    计算详细的性能指标,按运动类型分组
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
    
    # 加载运动类型信息
    with open(json_file, 'r') as f:
        all_data = json.load(f)
    
    motion_types = {}
    for frame_data in all_data:
        for obj in frame_data['objects']:
            motion_types[obj['id']] = obj.get('motion_type', 'unknown')
    
    print("🔍 Computing detailed metrics...\n")
    
    # 按运动类型分组统计
    metrics_by_type = {}
    
    with torch.no_grad():
        for i in range(len(dataset)):
            input_seq, target_seq = dataset[i]
            sample_info = dataset.samples[i]
            
            obj_id = sample_info['obj_id']
            motion_type = motion_types.get(obj_id, 'unknown')
            
            if motion_type not in metrics_by_type:
                metrics_by_type[motion_type] = {
                    'mse': [],
                    'mae': [],
                    'iou': []
                }
            
            # 预测
            input_tensor = input_seq.unsqueeze(0).to(device)
            predictions = model.predict_sequence(input_tensor, steps=target_seq.size(0))
            predictions = predictions.squeeze(0).cpu().numpy()
            target_np = target_seq.numpy()
            
            # MSE
            mse = np.mean((predictions - target_np) ** 2)
            metrics_by_type[motion_type]['mse'].append(mse)
            
            # MAE
            mae = np.mean(np.abs(predictions - target_np))
            metrics_by_type[motion_type]['mae'].append(mae)
            
            # IoU
            for pred_box, target_box in zip(predictions, target_np):
                iou = compute_iou(pred_box, target_box)
                metrics_by_type[motion_type]['iou'].append(iou)
    
    # 打印结果
    print("=" * 80)
    print("📊 Performance Metrics by Motion Type")
    print("=" * 80)
    
    for motion_type in sorted(metrics_by_type.keys()):
        metrics = metrics_by_type[motion_type]
        
        print(f"\n🎯 {motion_type.upper()}")
        print(f"  Samples: {len(metrics['mse'])}")
        print(f"  MSE:  Mean={np.mean(metrics['mse']):.2f}, Median={np.median(metrics['mse']):.2f}")
        print(f"  MAE:  Mean={np.mean(metrics['mae']):.2f}, Median={np.median(metrics['mae']):.2f}")
        print(f"  IoU:  Mean={np.mean(metrics['iou']):.3f}, Median={np.median(metrics['iou']):.3f}")
        print(f"        >0.5: {np.sum(np.array(metrics['iou']) > 0.5) / len(metrics['iou']) * 100:.1f}%")
    
    print("=" * 80)


def compute_iou(box1, box2):
    """计算IoU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 并集
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🎨 Enhanced Visualization: Ground Truth vs LSTM Prediction")
    print("=" * 80 + "\n")
    
    json_file = "./path2_output_corrected/stage1/motion_sequence.json"
    model_file = "./path2_output_corrected/stage2/best_lstm_tracker.pth"
    
    if not Path(json_file).exists():
        print(f"⚠️  Data file not found: {json_file}")
        return
    
    if not Path(model_file).exists():
        print(f"⚠️  Model file not found: {model_file}")
        return
    
    # 1. 可视化对比
    print("1️⃣  Visualizing Ground Truth vs Prediction...")
    visualize_prediction_vs_ground_truth(model_file, json_file, num_samples=3)
    
    # 2. 详细指标
    print("\n2️⃣  Computing detailed metrics by motion type...")
    compute_detailed_metrics(model_file, json_file)
    
    print("\n✅ All visualizations completed!")


if __name__ == "__main__":
    main()

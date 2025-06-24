import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
from torch.utils.data import Dataset, DataLoader

# 导入自定义模块
from models.simpletm_fall_detector import SimpleTM_FallDetector, Config

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SimpleTM跌倒检测模型预测')
    
    # 数据参数
    parser.add_argument('--input_file', type=str, required=True,
                      help='输入文件路径（CSV格式）')
    parser.add_argument('--use_splits', action='store_true',
                      help='是否使用预分割的数据集')
    parser.add_argument('--splits_dir', type=str, default='../KFall Dataset/time_series_splits',
                      help='预分割数据集目录')
    parser.add_argument('--window_size', type=int, default=128,
                      help='窗口大小')
    parser.add_argument('--stride', type=int, default=32,
                      help='滑动窗口步长')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth',
                      help='模型权重路径')
    parser.add_argument('--scaler_path', type=str, default='./checkpoints/scaler.joblib',
                      help='标准化器路径')
    parser.add_argument('--d_model', type=int, default=256, 
                      help='模型维度')
    parser.add_argument('--e_layers', type=int, default=1, 
                      help='编码器层数')
    parser.add_argument('--alpha', type=float, default=0.5, 
                      help='几何注意力中的平衡系数')
    parser.add_argument('--kernel_size', type=int, default=4, 
                      help='小波变换核大小')
    parser.add_argument('--m', type=int, default=2, 
                      help='小波分解级别')
    parser.add_argument('--dropout', type=float, default=0.2, 
                      help='Dropout率')
    parser.add_argument('--num_classes', type=int, default=2, 
                      help='类别数')
    
    # 输出参数
    parser.add_argument('--output_file', type=str, default='./results/predictions.csv',
                      help='预测结果输出路径')
    parser.add_argument('--visualize', action='store_true',
                      help='是否可视化预测结果')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='推理设备')
    
    return parser.parse_args()

class TestDataset(Dataset):
    """用于预测的数据集"""
    def __init__(self, X):
        self.X = torch.FloatTensor(X)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]

def preprocess_data(data, window_size=128, stride=32, scaler=None):
    """预处理输入数据"""
    # 选择传感器数据列（排除时间戳、标签等）
    sensor_columns = [col for col in data.columns 
                     if col not in ['timestamp', 'label', 'activity', 'subject']]
    
    # 填充缺失值
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    # 标准化数据（如果提供了scaler）
    if scaler:
        data[sensor_columns] = scaler.transform(data[sensor_columns])
    
    # 创建滑动窗口
    X_windows = []
    timestamps = []
    
    for i in range(0, len(data) - window_size + 1, stride):
        window = data[sensor_columns].iloc[i:i+window_size].values
        X_windows.append(window)
        
        # 记录窗口的中间时间点（如果有时间戳）
        if 'timestamp' in data.columns:
            mid_point = i + window_size // 2
            timestamps.append(data['timestamp'].iloc[mid_point])
    
    X = np.array(X_windows)
    
    # 转换为模型需要的格式：[n_samples, n_features, seq_len]
    X = np.transpose(X, (0, 2, 1))
    
    return X, timestamps

def visualize_predictions(data, timestamps, predictions, output_file):
    """可视化预测结果"""
    plt.figure(figsize=(12, 6))
    
    # 如果存在加速度数据，绘制加速度波形
    accel_cols = [col for col in data.columns if 'acc' in col.lower()]
    if accel_cols:
        for col in accel_cols[:3]:  # 最多绘制3个加速度通道
            plt.plot(data.index, data[col], alpha=0.5, label=col)
    
    # 绘制预测结果
    fall_indices = np.where(predictions == 1)[0]
    if len(fall_indices) > 0:
        fall_times = [timestamps[i] if timestamps else i for i in fall_indices]
        for t in fall_times:
            plt.axvline(x=t, color='red', linestyle='--', alpha=0.7)
    
    plt.title('Fall Detection Predictions')
    plt.xlabel('Time')
    plt.ylabel('Sensor Values')
    plt.legend()
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_file.replace('.csv', '.png'))
    plt.close()

def main():
    args = parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件 {args.input_file} 不存在!")
        return
    
    print("SimpleTM跌倒检测预测脚本")
    print(f"输入文件: {args.input_file}")
    print(f"模型路径: {args.model_path}")
    print(f"设备: {args.device}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    try:
        data = pd.read_csv(args.input_file)
        print(f"数据加载成功，共 {len(data)} 条记录")
    except Exception as e:
        print(f"加载数据错误: {e}")
        return
    
    # 加载标准化器（如果存在）
    scaler = None
    if os.path.exists(args.scaler_path):
        print(f"加载标准化器: {args.scaler_path}")
        try:
            scaler = joblib.load(args.scaler_path)
            print("标准化器加载成功")
        except Exception as e:
            print(f"加载标准化器错误: {e}")
            print("将使用未标准化的数据进行预测")
    
    # 预处理数据
    print("预处理数据...")
    try:
        X, timestamps = preprocess_data(data, args.window_size, args.stride, scaler)
        print(f"预处理完成，生成了 {len(X)} 个窗口样本")
    except Exception as e:
        print(f"预处理数据错误: {e}")
        return
    
    if len(X) == 0:
        print("错误: 预处理后没有窗口样本，请检查数据或调整窗口参数")
        return
    
    # 创建数据加载器
    test_dataset = TestDataset(X)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 初始化模型
    print("初始化模型...")
    model_config = Config(
        seq_len=args.window_size,
        d_model=args.d_model,
        dropout=args.dropout,
        output_attention=True,
        use_norm=True,
        geomattn_dropout=args.dropout,
        alpha=args.alpha,
        kernel_size=args.kernel_size,
        embed='fixed',
        freq='h',
        factor=5,
        requires_grad=False,
        wv='db2',
        m=args.m,
        dec_in=X.shape[1],  # 输入特征维度
        e_layers=args.e_layers,
        d_ff=args.d_model * 4,
        activation='gelu',
        num_classes=args.num_classes
    )
    
    try:
        model = SimpleTM_FallDetector(model_config)
        model.to(args.device)
        print(f"模型初始化成功，输入特征数: {X.shape[1]}")
    except Exception as e:
        print(f"初始化模型错误: {e}")
        return
    
    # 加载模型权重
    print(f"加载模型权重: {args.model_path}")
    if os.path.exists(args.model_path):
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=args.device))
            print("模型权重加载成功")
        except Exception as e:
            print(f"加载模型权重错误: {e}")
            return
    else:
        print(f"错误: 模型文件 {args.model_path} 不存在!")
        return
    
    # 进行预测
    print("开始预测...")
    model.eval()
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, x in enumerate(test_loader):
            x = x.to(args.device)
            
            # 前向传播
            try:
                outputs, _ = model(x)
                
                # 获取预测概率和类别
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # 保存预测结果
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
            except Exception as e:
                print(f"批次 {batch_idx} 预测错误: {e}")
                continue
    
    # 处理结果
    print("处理预测结果...")
    try:
        # 创建结果DataFrame
        results = pd.DataFrame()
        
        # 如果有时间戳，添加到结果中
        if timestamps:
            results['timestamp'] = timestamps
        else:
            results['window_idx'] = range(len(all_preds))
        
        results['prediction'] = all_preds
        results['confidence'] = [prob[pred] for prob, pred in zip(all_probs, all_preds)]
        
        # 添加标签名称
        class_names = ['normal', 'fall'] if args.num_classes == 2 else [f'class {i}' for i in range(args.num_classes)]
        results['prediction_label'] = [class_names[pred] for pred in all_preds]
        
        # 计算统计信息
        fall_count = sum(all_preds)
        total_count = len(all_preds)
        fall_ratio = fall_count / total_count if total_count > 0 else 0
        
        # 将预测结果保存为CSV
        results.to_csv(args.output_file, index=False, encoding='utf-8')
        print(f"预测结果已保存至: {args.output_file}")
        print(f"总窗口数: {total_count}, 预测为跌倒的窗口数: {fall_count}, 跌倒比例: {fall_ratio:.2%}")
        
        # 可视化预测结果
        if args.visualize:
            visualization_file = args.output_file.replace('.csv', '.png')
            try:
                visualize_predictions(data, timestamps or range(len(all_preds)), all_preds, visualization_file)
                print(f"可视化结果已保存至: {visualization_file}")
            except Exception as e:
                print(f"可视化结果时出错: {e}")
    
    except Exception as e:
        print(f"处理结果时出错: {e}")
    
    print("预测完成！")

if __name__ == "__main__":
    main()
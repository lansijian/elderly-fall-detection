import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

# 导入自定义模块
from models.simpletm_fall_detector import SimpleTM_FallDetector, Config
from data_processors.data_processor import prepare_data_for_simpleTM, load_from_time_series_splits

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SimpleTM跌倒检测模型评估')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='../KFall Dataset/processed_data/kfall_processed_data.csv',
                      help='传感器数据路径')
    parser.add_argument('--use_splits', action='store_true',
                      help='是否使用预分割的数据集')
    parser.add_argument('--splits_dir', type=str, default='../KFall Dataset/time_series_splits',
                      help='预分割数据集目录')
    parser.add_argument('--window_size', type=int, default=128, 
                      help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=32, 
                      help='滑动窗口步长')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth',
                      help='模型权重路径')
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
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=64, 
                      help='批次大小')
    parser.add_argument('--results_dir', type=str, default='./evaluation_results',
                      help='结果保存目录')
    parser.add_argument('--save_attention', action='store_true',
                      help='是否保存注意力图')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='评估设备')
    
    return parser.parse_args()

def visualize_attention(attention, save_path, window_size):
    """可视化注意力权重"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, cmap='viridis')
    plt.title('Attention Weights Heatmap')
    plt.xlabel('Sequence Position')
    plt.ylabel('Sequence Position')
    
    # 在大型序列中减少坐标轴刻度数量
    if window_size > 20:
        step = max(1, window_size // 10)
        plt.xticks(np.arange(0, window_size, step))
        plt.yticks(np.arange(0, window_size, step))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def save_confusion_matrix(y_true, y_pred, save_path, title='Confusion Matrix', class_names=None):
    """绘制并保存混淆矩阵"""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    
    # 如果没有提供类名，则使用默认值
    if class_names is None:
        n_classes = len(np.unique(y_true))
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    # 使用seaborn生成热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, test_loader, criterion, device, save_attention=False, attention_dir=None):
    """评估模型在测试集上的性能"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_attentions = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output, attns = model(data)
            
            # 计算损失
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # 获取预测结果
            preds = torch.argmax(output, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())
            
            # 保存注意力权重（如果需要）
            if save_attention and attention_dir:
                # 假设attns是一个列表，每个元素是一层的注意力权重
                for layer_idx, attn in enumerate(attns):
                    if isinstance(attn, tuple):
                        attn = attn[0]  # 如果是元组，取第一个元素（注意力矩阵）
                    # 对每个批次样本可视化
                    for sample_idx in range(min(3, data.shape[0])):  # 限制每批次只可视化前3个样本
                        if hasattr(attn, 'shape') and len(attn.shape) >= 4:
                            # 如果是多头注意力，取平均
                            attn_map = attn[sample_idx].mean(0).cpu().numpy()
                            save_path = os.path.join(attention_dir, f'attn_batch{batch_idx}_sample{sample_idx}_layer{layer_idx}.png')
                            visualize_attention(attn_map, save_path, data.shape[2])
    
    # 计算平均损失和评估指标
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels

def main():
    """主函数"""
    args = parse_args()
    
    # 创建结果保存目录
    os.makedirs(args.results_dir, exist_ok=True)
    if args.save_attention:
        attention_dir = os.path.join(args.results_dir, 'attention_maps')
        os.makedirs(attention_dir, exist_ok=True)
    else:
        attention_dir = None
    
    # 准备数据
    print("正在加载和准备数据...")
    
    if args.use_splits:
        # 使用预分割的数据集
        _, _, test_loader, scaler = load_from_time_series_splits(
            data_dir=args.splits_dir,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            normalize=True
        )
    else:
        # 使用完整数据集并进行分割
        _, _, test_loader, scaler = prepare_data_for_simpleTM(
            data_path=args.data_path,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            normalize=True
        )
    
    print(f"数据准备完成。测试批次数: {len(test_loader)}")
    
    # 获取输入特征维度
    sample_batch, _ = next(iter(test_loader))
    input_features = sample_batch.shape[1]  # [batch_size, n_features, seq_len]
    
    # 初始化模型配置
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
        dec_in=input_features,
        e_layers=args.e_layers,
        d_ff=args.d_model * 4,
        activation='gelu',
        num_classes=args.num_classes
    )
    
    # 初始化模型
    model = SimpleTM_FallDetector(model_config)
    model.to(args.device)
    
    # 加载模型权重
    print(f"加载模型权重: {args.model_path}")
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    else:
        print(f"错误: 模型文件 {args.model_path} 不存在!")
        return
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 评估模型
    print("开始评估模型...")
    test_loss, accuracy, precision, recall, f1, test_preds, test_labels = evaluate_model(
        model, test_loader, criterion, args.device, args.save_attention, attention_dir)
    
    # 打印评估结果
    print("\n测试集评估结果:")
    print(f"损失: {test_loss:.4f}")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 生成并保存混淆矩阵
    cm_path = os.path.join(args.results_dir, 'confusion_matrix.png')
    class_names = ['Normal', 'Fall'] if args.num_classes == 2 else [f'Class {i}' for i in range(args.num_classes)]
    save_confusion_matrix(test_labels, test_preds, cm_path, title='Test Set Confusion Matrix', class_names=class_names)
    print(f"混淆矩阵已保存至 {cm_path}")
    
    # 生成并保存分类报告
    report = classification_report(test_labels, test_preds, target_names=class_names, digits=4)
    report_path = os.path.join(args.results_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("SimpleTM Fall Detection Model Evaluation Report\n")
        f.write("==========================\n\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Window Size: {args.window_size}\n")
        f.write(f"Model Dimension: {args.d_model}\n")
        f.write(f"Encoder Layers: {args.e_layers}\n")
        f.write(f"Geometric Attention Alpha: {args.alpha}\n")
        f.write(f"Wavelet Decomposition Level (m): {args.m}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"分类报告已保存至 {report_path}")
    
    # 分析预测时间点与真实跌倒时间点的差异
    analyze_timing = False  # 这需要有时间戳信息
    if analyze_timing:
        # 此处添加分析代码
        pass
    
    print("评估完成!")

if __name__ == "__main__":
    main() 
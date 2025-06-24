import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 导入我们的自定义模块
from models.simpletm_fall_detector import SimpleTM_FallDetector, Config
from data_processors.data_processor import prepare_data_for_simpleTM

def parse_args():
    parser = argparse.ArgumentParser(description='SimpleTM跌倒检测模型训练')
    
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
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64, 
                      help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, 
                      help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, 
                      help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                      help='权重衰减')
    parser.add_argument('--patience', type=int, default=10, 
                      help='早停耐心值')
    parser.add_argument('--seed', type=int, default=42, 
                      help='随机种子')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=256, 
                      help='模型维度')
    parser.add_argument('--dropout', type=float, default=0.2, 
                      help='Dropout率')
    parser.add_argument('--e_layers', type=int, default=1, 
                      help='编码器层数')
    parser.add_argument('--alpha', type=float, default=0.5, 
                      help='几何注意力中的平衡系数')
    parser.add_argument('--kernel_size', type=int, default=4, 
                      help='小波变换核大小')
    parser.add_argument('--m', type=int, default=2, 
                      help='小波分解级别')
    parser.add_argument('--num_classes', type=int, default=2, 
                      help='类别数')
    
    # 输出参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints', 
                      help='模型保存目录')
    parser.add_argument('--results_dir', type=str, default='./results', 
                      help='结果保存目录')
    
    # GPU参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='训练设备')
    
    return parser.parse_args()

def set_seed(seed):
    """设置随机种子以确保可重现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output, _ = model(data)
        
        # 计算损失
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        
        # 存储预测结果和标签（用于计算指标）
        preds = torch.argmax(output, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(target.cpu().numpy())
    
    # 计算训练指标
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary' if len(set(all_labels)) <= 2 else 'weighted')
    recall = recall_score(all_labels, all_preds, average='binary' if len(set(all_labels)) <= 2 else 'weighted')
    f1 = f1_score(all_labels, all_preds, average='binary' if len(set(all_labels)) <= 2 else 'weighted')
    
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss, acc, precision, recall, f1

def evaluate(model, val_loader, criterion, device):
    """在验证集或测试集上评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output, _ = model(data)
            
            # 计算损失
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # 存储预测结果和标签
            preds = torch.argmax(output, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())
    
    # 计算验证指标
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary' if len(set(all_labels)) <= 2 else 'weighted')
    recall = recall_score(all_labels, all_preds, average='binary' if len(set(all_labels)) <= 2 else 'weighted')
    f1 = f1_score(all_labels, all_preds, average='binary' if len(set(all_labels)) <= 2 else 'weighted')
    
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, acc, precision, recall, f1, all_preds, all_labels

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

def plot_training_history(history, save_path):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制损失曲线
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # 绘制准确率曲线
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_title('Accuracy Curve')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # 绘制F1分数曲线
    axes[1, 0].plot(history['train_f1'], label='Train')
    axes[1, 0].plot(history['val_f1'], label='Validation')
    axes[1, 0].set_title('F1 Score Curve')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    
    # 绘制召回率曲线
    axes[1, 1].plot(history['train_recall'], label='Train')
    axes[1, 1].plot(history['val_recall'], label='Validation')
    axes[1, 1].set_title('Recall Curve')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def main():
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 准备数据
    print("正在加载和准备数据...")
    
    if args.use_splits:
        # 使用预分割的数据集
        from data_processors.data_processor import load_from_time_series_splits
        train_loader, val_loader, test_loader, scaler = load_from_time_series_splits(
            data_dir=args.splits_dir,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            normalize=True
        )
    else:
        # 使用完整数据集并进行分割
        from data_processors.data_processor import prepare_data_for_simpleTM
        train_loader, val_loader, test_loader, scaler = prepare_data_for_simpleTM(
            data_path=args.data_path,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            normalize=True
        )
    
    print(f"数据准备完成。训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}, 测试批次: {len(test_loader)}")
    
    # 获取输入特征维度
    sample_batch, _ = next(iter(train_loader))
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
    
    # 定义损失函数、优化器和学习率调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 修复ReduceLROnPlateau参数问题，检查PyTorch版本并适配
    try:
        # 尝试使用所有参数
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, 
                                     verbose=True, threshold=0.0001, threshold_mode='rel')
    except TypeError as e:
        print(f"ReduceLROnPlateau参数错误: {e}")
        print("尝试使用兼容模式创建学习率调度器...")
        try:
            # 去除可能引起问题的参数
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            print("使用兼容模式成功创建学习率调度器")
        except Exception as e2:
            print(f"创建学习率调度器失败: {e2}")
            print("将使用固定学习率训练")
            scheduler = None
    
    # 训练历史记录
    history = {
        'train_loss': [], 'train_acc': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    # 早停变量
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    patience_counter = 0
    
    # 开始训练
    print("开始训练...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # 训练
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )
        
        # 验证
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(
            model, val_loader, criterion, args.device
        )
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录训练历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)
        history['train_f1'].append(train_f1)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        # 检查是否为最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"Epoch {epoch}：保存了新的最佳模型")
        else:
            patience_counter += 1
        
        # 打印当前训练状态
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch}/{args.epochs} | 耗时: {epoch_time:.2f}s")
        print(f"训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}, 精确率: {train_precision:.4f}, 召回率: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"验证集 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}, 精确率: {val_precision:.4f}, 召回率: {val_recall:.4f}, F1: {val_f1:.4f}")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 80)
        
        # 检查早停
        if patience_counter >= args.patience:
            print(f"早停 - {args.patience} 个epoch内验证损失没有改善")
            break
    
    # 训练结束，计算总耗时
    total_time = time.time() - start_time
    print(f"训练完成！总耗时: {total_time:.2f}秒")
    
    # 绘制训练历史
    history_plot_path = os.path.join(args.results_dir, 'training_history.png')
    plot_training_history(history, history_plot_path)
    print(f"训练历史已保存至 {history_plot_path}")
    
    # 加载最佳模型并在测试集上评估
    print("在测试集上评估最佳模型...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_precision, test_recall, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, args.device
    )
    
    # 打印测试结果
    print("\n测试集评估结果:")
    print(f"损失: {test_loss:.4f}")
    print(f"准确率: {test_acc:.4f}")
    print(f"精确率: {test_precision:.4f}")
    print(f"召回率: {test_recall:.4f}")
    print(f"F1分数: {test_f1:.4f}")
    
    # 生成并保存混淆矩阵
    cm_path = os.path.join(args.results_dir, 'confusion_matrix.png')
    class_names = ['Normal', 'Fall'] if args.num_classes == 2 else [f'Class {i}' for i in range(args.num_classes)]
    save_confusion_matrix(test_labels, test_preds, cm_path, title='Test Set Confusion Matrix', class_names=class_names)
    print(f"混淆矩阵已保存至 {cm_path}")
    
    # 保存详细分类报告
    report = classification_report(test_labels, test_preds, target_names=class_names, digits=4)
    report_path = os.path.join(args.results_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("SimpleTM Fall Detection Model Evaluation Report\n")
        f.write("==========================\n\n")
        f.write(f"Window Size: {args.window_size}\n")
        f.write(f"Model Dimension: {args.d_model}\n")
        f.write(f"Encoder Layers: {args.e_layers}\n")
        f.write(f"Geometric Attention Alpha: {args.alpha}\n")
        f.write(f"Wavelet Decomposition Level (m): {args.m}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"分类报告已保存至 {report_path}")
    print("训练和评估过程全部完成！")

if __name__ == "__main__":
    main() 
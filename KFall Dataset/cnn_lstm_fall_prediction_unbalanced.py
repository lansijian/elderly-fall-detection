import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 强制使用英文标签，避免中文显示问题
USE_ENGLISH = True
print("使用英文标签显示所有图表")

# 定义超参数
WINDOW_SIZE = 100  # 滑动窗口大小（使用过去100个时间点预测）
STRIDE = 20        # 滑动窗口步长
BATCH_SIZE = 64    # 批处理大小
EPOCHS = 50        # 训练轮数
LEARNING_RATE = 0.0005  # 学习率
WEIGHT_DECAY = 1e-5     # L2正则化参数
HIDDEN_SIZE = 128  # LSTM隐藏层大小
NUM_LAYERS = 2     # LSTM层数
DROPOUT = 0.3      # Dropout比例
PREDICTION_HORIZON = 10  # 预测未来多少个时间点
PATIENCE = 10      # 早停耐心值

# CNN特定参数
CNN_FILTERS = [32, 64, 128]  # 每层卷积的滤波器数量
KERNEL_SIZES = [5, 3, 3]    # 卷积核大小
POOL_SIZES = [2, 2, 2]      # 池化大小

# 传感器特征列
SENSOR_COLS = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'EulerX', 'EulerY', 'EulerZ']

class TimeSeriesDataset(Dataset):
    # 时间序列数据集，使用滑动窗口生成样本
    
    def __init__(self, csv_file, window_size=WINDOW_SIZE, stride=STRIDE, prediction_horizon=PREDICTION_HORIZON):
        """
        初始化数据集
        
        参数:
            csv_file: CSV文件路径
            window_size: 滑动窗口大小
            stride: 滑动步长
            prediction_horizon: 预测未来时间点的数量
        """
        # 初始化数据集参数
        print(f"Loading data: {csv_file}")  # 加载数据
        self.df = pd.read_csv(csv_file)
        self.window_size = window_size
        self.stride = stride
        self.prediction_horizon = prediction_horizon
        
        # 准备数据
        self.prepare_data()
        
    def prepare_data(self):
        # 准备数据，生成滑动窗口样本
        self.samples = []
        
        # 按参与者、任务和试验分组处理
        groups = self.df.groupby(['participant_id', 'task_id', 'trial_id'])
        print(f"Dataset contains {len(groups)} sequences")  # 数据集包含的序列数
        
        for (participant, task, trial), group in groups:
            # 确保数据按帧计数器排序
            group = group.sort_values('FrameCounter')
            
            # 如果数据量不足一个窗口+预测范围，跳过
            if len(group) < self.window_size + self.prediction_horizon:
                continue
            
            # 应用滑动窗口
            for i in range(0, len(group) - self.window_size - self.prediction_horizon + 1, self.stride):
                # 输入窗口
                window = group.iloc[i:i + self.window_size]
                
                # 预测目标窗口
                target_window = group.iloc[i + self.window_size:i + self.window_size + self.prediction_horizon]
                
                # 提取特征和标签
                features = window[SENSOR_COLS].values
                
                # 如果预测窗口中有任何一帧是跌倒，则标记为跌倒风险
                target = 1 if target_window['label'].sum() > 0 else 0
                
                # 存储样本
                self.samples.append((features, target))
        
        print(f"Generated {len(self.samples)} samples")  # 生成的样本数
        
        # 检查标签分布
        labels = [sample[1] for sample in self.samples]
        label_counts = pd.Series(labels).value_counts()
        print(f"Label distribution:")  # 标签分布
        for label, count in label_counts.items():
            print(f"  Label {label}: {count} samples ({count/len(labels):.1%})")
    
    def __len__(self):
        # 返回数据集大小
        return len(self.samples)
    
    def __getitem__(self, idx):
        # 获取样本
        features, target = self.samples[idx]
        
        # 数据增强 - 随机添加少量噪声（仅在训练时）
        if hasattr(self, 'is_train') and self.is_train:
            noise = np.random.normal(0, 0.01, features.shape)
            features = features + noise

        # 转换为PyTorch张量
        features_tensor = torch.FloatTensor(features)
        target_tensor = torch.FloatTensor([target])
        
        return features_tensor, target_tensor
    
    def set_train_mode(self, is_train=True):
        """设置是否为训练模式，用于数据增强"""
        self.is_train = is_train

class CNNLSTM(nn.Module):
    """
    CNN+LSTM混合模型：先使用1D-CNN提取局部模式特征，再用LSTM处理时序依赖关系
    """
    def __init__(self, input_size=len(SENSOR_COLS), hidden_size=HIDDEN_SIZE, 
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        """
        初始化混合模型
        
        参数:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比例
        """
        super(CNNLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        # 批标准化层（输入数据）
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # 1D-CNN部分 - 提取局部特征
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.act_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # 第一层卷积处理原始输入
        self.conv_layers.append(nn.Conv1d(input_size, CNN_FILTERS[0], kernel_size=KERNEL_SIZES[0], padding=KERNEL_SIZES[0]//2))
        self.bn_layers.append(nn.BatchNorm1d(CNN_FILTERS[0]))
        self.act_layers.append(nn.ReLU())
        self.pool_layers.append(nn.MaxPool1d(POOL_SIZES[0]))
        self.dropout_layers.append(nn.Dropout(dropout * 0.5))  # 较低的dropout
        
        # 后续卷积层
        for i in range(1, len(CNN_FILTERS)):
            self.conv_layers.append(nn.Conv1d(CNN_FILTERS[i-1], CNN_FILTERS[i], kernel_size=KERNEL_SIZES[i], padding=KERNEL_SIZES[i]//2))
            self.bn_layers.append(nn.BatchNorm1d(CNN_FILTERS[i]))
            self.act_layers.append(nn.ReLU())
            self.pool_layers.append(nn.MaxPool1d(POOL_SIZES[i]))
            self.dropout_layers.append(nn.Dropout(dropout * (0.5 + i * 0.1)))  # 逐渐增加dropout
        
        # 计算CNN输出后的序列长度
        self.cnn_out_size = WINDOW_SIZE
        for p in POOL_SIZES:
            self.cnn_out_size = self.cnn_out_size // p
            
        # LSTM层 - 学习时序依赖关系
        self.lstm = nn.LSTM(
            input_size=CNN_FILTERS[-1],  # CNN最后一层的滤波器数量
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # 单向LSTM
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
            # 不含sigmoid，因为使用BCEWithLogitsLoss
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, seq_len, input_size)
            
        返回:
            输出张量，形状为 (batch_size, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # 转换输入形状以适应卷积层 (batch_size, input_size, seq_len)
        x = x.permute(0, 2, 1)
        
        # 应用输入批标准化
        x = self.input_bn(x)
        
        # 应用CNN层
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)
            x = self.act_layers[i](x)
            x = self.pool_layers[i](x)
            x = self.dropout_layers[i](x)
        
        # 转换形状以适应LSTM (batch_size, seq_len, channels)
        x = x.permute(0, 2, 1)
        
        # 应用LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size)
        
        # 应用注意力机制
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_size)
        
        # 全连接层
        output = self.fc(context_vector)  # (batch_size, 1)
        
        return output

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=PATIENCE, delta=0.001, mode='max', verbose=True):
        """
        初始化早停
        
        参数:
            patience: 在多少个epoch没有改善后停止训练
            delta: 最小变化阈值，小于此值视为没有改善
            mode: 'min'表示监视指标越小越好，'max'表示监视指标越大越好
            verbose: 是否打印早停信息
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_best = 0 if mode == 'max' else float('inf')
    
    def __call__(self, current_score, model, model_path):
        """
        调用早停检查
        
        参数:
            current_score: 当前监视指标
            model: 当前模型
            model_path: 保存模型路径
        """
        if self.mode == 'max':
            score = current_score
        else:
            score = -current_score
            
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(current_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(current_score, model, model_path)
            self.counter = 0
            
    def save_checkpoint(self, current_score, model, model_path):
        """保存最佳模型"""
        if self.verbose:
            print(f'模型性能提升 ({self.val_best:.4f} --> {current_score:.4f}). 保存模型...')
        torch.save(model.state_dict(), model_path)
        self.val_best = current_score

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=EPOCHS):
    """
    训练模型
    
    参数:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        epochs: 训练轮数
        
    返回:
        训练历史
    """
    # 训练模型
    model.to(device)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_auc': []
    }
    
    # 初始化早停
    early_stopping = EarlyStopping(patience=PATIENCE, mode='max')
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        # 输出当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{epochs} (LR: {current_lr:.6f})")
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 更新参数
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 前向传播
                outputs = model(inputs)
                
                # 计算损失
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                
                # sigmoid激活用于二分类预测
                preds = torch.sigmoid(outputs)
                
                # 存储预测和目标
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 计算评估指标
        val_preds = np.array(val_preds).flatten()
        val_targets = np.array(val_targets).flatten()
        
        val_preds_binary = (val_preds > 0.5).astype(int)
        val_accuracy = np.mean(val_preds_binary == val_targets)
        
        # 计算AUC
        try:
            val_auc = roc_auc_score(val_targets, val_preds)
        except:
            val_auc = 0.5  # 如果只有一个类别，无法计算AUC
        
        # 存储历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_auc'].append(val_auc)
        
        # 打印进度
        print(f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"Val Accuracy: {val_accuracy:.4f} - "
              f"Val AUC: {val_auc:.4f}")
        
        # 早停检查
        early_stopping(val_auc, model, 'best_model_cnn_lstm_unbalanced.pth')
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return history

def evaluate_model(model, test_loader, device):
    """评估模型在测试集上的表现"""
    model.eval()
    all_targets = []
    all_predictions = []
    all_probs = []  # 存储模型输出的原始概率
    
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            
            # 使用sigmoid将输出转换为概率
            probs = torch.sigmoid(outputs)
            
            # 使用我们找到的最佳阈值0.84进行分类，而不是默认的0.5
            predicted = (probs > 0.84).float()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()) # 收集概率
            
    # 转换为numpy数组
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    
    # 打印分类报告
    print("\n" + "="*30)
    print("Classification Report:")
    print("="*30)
    print(classification_report(all_targets, all_predictions, target_names=['Non-Fall', 'Fall']))
    
    # 计算并打印AUC
    auc_score = roc_auc_score(all_targets, all_probs)
    print(f"ROC AUC Score: {auc_score:.4f}")

    # 计算并打印Average Precision Score (AP)，它总结了P-R曲线
    ap_score = average_precision_score(all_targets, all_probs)
    print(f"Average Precision (AP) Score: {ap_score:.4f}")
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Fall', 'Fall'], 
                yticklabels=['Non-Fall', 'Fall'])
    plt.title('Confusion Matrix for CNN-LSTM (Unbalanced)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_cnn_lstm_unbalanced.png')
    print("\nConfusion matrix saved as 'confusion_matrix_cnn_lstm_unbalanced.png'")
    plt.show()

    # 绘制P-R曲线
    precision, recall, _ = precision_recall_curve(all_targets, all_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f'CNN-LSTM (AP = {ap_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for CNN-LSTM (Unbalanced)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'pr_curve_cnn_lstm_unbalanced.png')
    print("P-R curve saved as 'pr_curve_cnn_lstm_unbalanced.png'")
    plt.show()

    return all_targets, all_predictions, all_probs

def plot_training_history(history):
    """
    绘制训练历史
    
    参数:
        history: 训练历史
    """
    # 绘制训练历史
    plt.figure(figsize=(12, 4))
    
    # 使用英文标签
    loss_title = 'Training and Validation Loss (CNN-LSTM - Unbalanced)'  # 训练和验证损失 (CNN-LSTM - 未平衡)
    metrics_title = 'Validation Accuracy and AUC (CNN-LSTM - Unbalanced)'  # 验证准确率和AUC (CNN-LSTM - 未平衡)
    train_loss_label = 'Training Loss'  # 训练损失
    val_loss_label = 'Validation Loss'  # 验证损失
    val_acc_label = 'Validation Accuracy'  # 验证准确率
    val_auc_label = 'Validation AUC'  # 验证AUC
    x_label = 'Epoch'  # 轮次
    y_loss_label = 'Loss'  # 损失
    y_metrics_label = 'Score'  # 评分
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label=train_loss_label)
    plt.plot(history['val_loss'], label=val_loss_label)
    plt.xlabel(x_label)
    plt.ylabel(y_loss_label)
    plt.title(loss_title)
    plt.legend()
    
    # 绘制AUC
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label=val_acc_label)
    plt.plot(history['val_auc'], label=val_auc_label)
    plt.xlabel(x_label)
    plt.ylabel(y_metrics_label)
    plt.title(metrics_title)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_cnn_lstm_unbalanced.png')

def main():
    """主函数"""
    # 主函数
    print("=== CNN-LSTM混合模型跌倒预测 (原始未平衡数据) ===\n")  # CNN-LSTM混合模型跌倒预测（原始未平衡数据）
    
    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")  # 使用设备
    
    # 数据路径
    script_dir = Path(__file__).parent
    train_path = script_dir / 'time_series_splits' / 'train.csv'
    val_path = script_dir / 'time_series_splits' / 'val.csv'
    test_path = script_dir / 'time_series_splits' / 'test.csv'
    
    # 创建数据集
    train_dataset = TimeSeriesDataset(train_path)
    val_dataset = TimeSeriesDataset(val_path)
    test_dataset = TimeSeriesDataset(test_path)
    
    # 设置训练数据集为训练模式（启用数据增强）
    train_dataset.set_train_mode(True)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 创建模型
    model = CNNLSTM()
    print(f"\nModel structure:")  # 模型结构
    print(model)
    
    # 定义损失函数和优化器
    # 为了处理数据不平衡，使用带权重的BCE损失
    pos_weight = torch.tensor([5.0])  # 增加正样本权重，因为未平衡数据集中正样本更少
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 使用Adam优化器，带L2正则化
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 学习率调度器，当验证损失不再下降时减小学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 训练模型
    print(f"\n开始模型训练...")  # 开始训练模型
    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device)
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model_cnn_lstm_unbalanced.pth'))
    
    # 评估模型
    print(f"\n评估最佳模型...")  # 评估最佳模型
    evaluation = evaluate_model(model, test_loader, device)
    
    print(f"\n=== 训练和评估完成 ===")  # 训练和评估完成
    print(f"最佳模型已保存为 best_model_cnn_lstm_unbalanced.pth")  # 最佳模型已保存
    print(f"混淆矩阵已保存为 confusion_matrix_cnn_lstm_unbalanced.png")  # 混淆矩阵已保存
    print(f"训练历史已保存为 training_history_cnn_lstm_unbalanced.png")  # 训练历史已保存

if __name__ == "__main__":
    main()
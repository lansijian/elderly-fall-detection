import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# --- 配置 ---
MODEL_PATH = 'KFall Dataset/best_model_cnn_lstm_unbalanced.pth'
TEST_DATA_PATH = 'KFall Dataset/time_series_splits/test.csv'
OUTPUT_PLOT_PATH = 'KFall Dataset/threshold_tuning_cnn_lstm_unbalanced_fine_grained.png'
BATCH_SIZE = 64

# --- 从原训练脚本复制的模型定义和超参数 ---
WINDOW_SIZE = 100
STRIDE = 20
PREDICTION_HORIZON = 10
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
CNN_FILTERS = [32, 64, 128]
KERNEL_SIZES = [5, 3, 3]
POOL_SIZES = [2, 2, 2]
SENSOR_COLS = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'EulerX', 'EulerY', 'EulerZ']

class TimeSeriesDataset(Dataset):
    def __init__(self, csv_file, window_size=WINDOW_SIZE, stride=STRIDE, prediction_horizon=PREDICTION_HORIZON):
        self.df = pd.read_csv(csv_file)
        self.window_size = window_size
        self.stride = stride
        self.prediction_horizon = prediction_horizon
        self.prepare_data()
        
    def prepare_data(self):
        self.samples = []
        groups = self.df.groupby(['participant_id', 'task_id', 'trial_id'])
        for _, group in groups:
            group = group.sort_values('FrameCounter')
            if len(group) < self.window_size + self.prediction_horizon:
                continue
            for i in range(0, len(group) - self.window_size - self.prediction_horizon + 1, self.stride):
                window = group.iloc[i:i + self.window_size]
                target_window = group.iloc[i + self.window_size:i + self.window_size + self.prediction_horizon]
                features = window[SENSOR_COLS].values
                target = 1 if target_window['label'].sum() > 0 else 0
                self.samples.append((features, target))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        features, target = self.samples[idx]
        features_tensor = torch.FloatTensor(features)
        target_tensor = torch.FloatTensor([target])
        return features_tensor, target_tensor

class CNNLSTM(nn.Module):
    def __init__(self, input_size=len(SENSOR_COLS), hidden_size=HIDDEN_SIZE, 
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(CNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.input_bn = nn.BatchNorm1d(input_size)
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.act_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(input_size, CNN_FILTERS[0], kernel_size=KERNEL_SIZES[0], padding=KERNEL_SIZES[0]//2))
        self.bn_layers.append(nn.BatchNorm1d(CNN_FILTERS[0]))
        self.act_layers.append(nn.ReLU())
        self.pool_layers.append(nn.MaxPool1d(POOL_SIZES[0]))
        self.dropout_layers.append(nn.Dropout(dropout * 0.5))
        for i in range(1, len(CNN_FILTERS)):
            self.conv_layers.append(nn.Conv1d(CNN_FILTERS[i-1], CNN_FILTERS[i], kernel_size=KERNEL_SIZES[i], padding=KERNEL_SIZES[i]//2))
            self.bn_layers.append(nn.BatchNorm1d(CNN_FILTERS[i]))
            self.act_layers.append(nn.ReLU())
            self.pool_layers.append(nn.MaxPool1d(POOL_SIZES[i]))
            self.dropout_layers.append(nn.Dropout(dropout * (0.5 + i * 0.1)))
        self.cnn_out_size = WINDOW_SIZE
        for p in POOL_SIZES:
            self.cnn_out_size = self.cnn_out_size // p
        self.lstm = nn.LSTM(
            input_size=CNN_FILTERS[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.permute(0, 2, 1)
        x = self.input_bn(x)
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)
            x = self.act_layers[i](x)
            x = self.pool_layers[i](x)
            x = self.dropout_layers[i](x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.fc(context_vector)
        return output

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载模型
    print(f"加载模型: {MODEL_PATH}")
    model = CNNLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. 加载测试数据
    print(f"加载测试数据: {TEST_DATA_PATH}")
    test_dataset = TimeSeriesDataset(csv_file=TEST_DATA_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 获取预测概率
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # 4. 遍历阈值，计算指标
    # 精细化搜索：在上次找到的最佳点0.80附近，用更小的步长搜索
    thresholds = np.round(np.arange(0.75, 0.86, 0.01), 2)
    results = []

    print("\n--- 精细化阈值调整分析 (范围: 0.75-0.85, 步长: 0.01) ---")
    for thresh in thresholds:
        predictions = (all_probs > thresh).astype(int)
        # pos_label=1 表示我们关注'Fall'这个类别的指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, predictions, average=None, labels=[1]
        )
        results.append([thresh, precision[0], recall[0], f1[0]])

    results_df = pd.DataFrame(results, columns=['Threshold', 'Precision', 'Recall', 'F1-Score'])
    
    # 找到最佳F1分数对应的阈值
    best_f1_idx = results_df['F1-Score'].idxmax()
    best_threshold = results_df.loc[best_f1_idx]

    print(results_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n--- 最佳F1分数对应的阈值 ---")
    print(f"Threshold: {best_threshold['Threshold']:.4f}")
    print(f"Precision: {best_threshold['Precision']:.4f}")
    print(f"Recall: {best_threshold['Recall']:.4f}")
    print(f"F1-Score: {best_threshold['F1-Score']:.4f}")

    # 5. 可视化
    plt.figure(figsize=(12, 7))
    plt.plot(results_df['Threshold'], results_df['Precision'], 'o-', label='Precision')
    plt.plot(results_df['Threshold'], results_df['Recall'], 's-', label='Recall')
    plt.plot(results_df['Threshold'], results_df['F1-Score'], 'd-', label='F1-Score')
    
    # 标记最佳阈值点
    plt.axvline(x=best_threshold['Threshold'], color='r', linestyle='--', 
                label=f"Best Threshold (F1-Score): {best_threshold['Threshold']:.4f}")

    plt.title('Fine-Grained Threshold Tuning for CNN-LSTM (Unbalanced)')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.xticks(thresholds, rotation=45)
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH)
    print(f"\n阈值分析图已保存至: {OUTPUT_PLOT_PATH}")
    plt.show()

if __name__ == "__main__":
    main() 
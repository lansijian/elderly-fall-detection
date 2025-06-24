import torch
import torch.nn as nn

# --- 全局超参数 ---
SENSOR_COLS = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'EulerX', 'EulerY', 'EulerZ']
WINDOW_SIZE = 100

# --- 模型特定超参数 ---
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT_REGULARIZED = 0.3
LSTM_DROPOUT_ORIGINAL = 0.2
CNN_FILTERS = [32, 64, 128]
CNN_KERNEL_SIZES = [5, 3, 3]
CNN_POOL_SIZES = [2, 2, 2]

# ==============================================================================
# Model: best_model.pth
# Source: KFall Dataset/fall_prediction_lstm.py
# ==============================================================================
class OriginalLSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, num_layers=2, dropout=0.2):
        super(OriginalLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
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
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.fc(context_vector)
        return output

# ==============================================================================
# Model: best_model_lstm_regularized.pth
# Source: KFall Dataset/lstm_fall_prediction_balanced_regularized.py
# ==============================================================================
class RegularizedLSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, num_layers=2, dropout=0.3):
        super(RegularizedLSTM, self).__init__()
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
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
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x_reshaped = x.reshape(batch_size * seq_len, -1)
        x_normalized = self.batch_norm(x_reshaped)
        x = x_normalized.reshape(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.fc(context_vector)
        return output

# ==============================================================================
# Model: best_model_gru.pth
# Source: KFall Dataset/gru_fall_prediction_balanced.py
# ==============================================================================
class StandardGRU(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, num_layers=2, dropout=0.2):
        super(StandardGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
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
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attention_weights = self.attention(gru_out)
        context_vector = torch.sum(attention_weights * gru_out, dim=1)
        output = self.fc(context_vector)
        return output

# ==============================================================================
# Model: best_model_bidirectional_gru.pth
# Source: KFall Dataset/bidirectional_gru_fall_prediction.py
# ==============================================================================
class BiGRU(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, num_layers=2, dropout=0.3):
        super(BiGRU, self).__init__()
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.Tanh(),
            nn.Linear(hidden_size * 2, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x_reshaped = x.reshape(batch_size * seq_len, -1)
        x_normalized = self.batch_norm(x_reshaped)
        x = x_normalized.reshape(batch_size, seq_len, -1)
        
        gru_out, _ = self.gru(x)
        attention_weights = self.attention(gru_out)
        context_vector = torch.sum(attention_weights * gru_out, dim=1)
        output = self.fc(context_vector)
        return output

# ==============================================================================
# Model: best_model_cnn_lstm.pth
# Source: KFall Dataset/cnn_lstm_fall_prediction.py
# ==============================================================================
class CNNLSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, num_layers=2, dropout=0.3):
        super(CNNLSTM, self).__init__()
        cnn_filters = [32, 64, 128]
        kernel_sizes = [5, 3, 3]
        pool_sizes = [2, 2, 2]
        
        self.input_bn = nn.BatchNorm1d(input_size)
        
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.act_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        self.conv_layers.append(nn.Conv1d(input_size, cnn_filters[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2))
        self.bn_layers.append(nn.BatchNorm1d(cnn_filters[0]))
        self.act_layers.append(nn.ReLU())
        self.pool_layers.append(nn.MaxPool1d(pool_sizes[0]))
        self.dropout_layers.append(nn.Dropout(dropout * 0.5))
        
        for i in range(1, len(cnn_filters)):
            self.conv_layers.append(nn.Conv1d(cnn_filters[i-1], cnn_filters[i], kernel_size=kernel_sizes[i], padding=kernel_sizes[i]//2))
            self.bn_layers.append(nn.BatchNorm1d(cnn_filters[i]))
            self.act_layers.append(nn.ReLU())
            self.pool_layers.append(nn.MaxPool1d(pool_sizes[i]))
            self.dropout_layers.append(nn.Dropout(dropout * (0.5 + i * 0.1)))
            
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
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
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x_reshaped = x.reshape(batch_size * seq_len, -1)
        x_normalized = self.input_bn(x_reshaped)
        x = x_normalized.reshape(batch_size, seq_len, -1)
        
        x = x.permute(0, 2, 1)
        
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

# ==============================================================================
# Model: best_model_cnn_gru.pth
# Source: KFall Dataset/cnn_gru_fall_prediction.py
# ==============================================================================
class CNNGRU(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, num_layers=2, dropout=0.3):
        super(CNNGRU, self).__init__()
        cnn_filters = [32, 64, 128]
        kernel_sizes = [5, 3, 3]
        pool_sizes = [2, 2, 2]

        self.input_bn = nn.BatchNorm1d(input_size)
        
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.act_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        self.conv_layers.append(nn.Conv1d(input_size, cnn_filters[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2))
        self.bn_layers.append(nn.BatchNorm1d(cnn_filters[0]))
        self.act_layers.append(nn.ReLU())
        self.pool_layers.append(nn.MaxPool1d(pool_sizes[0]))
        self.dropout_layers.append(nn.Dropout(dropout * 0.5))

        for i in range(1, len(cnn_filters)):
            self.conv_layers.append(nn.Conv1d(cnn_filters[i-1], cnn_filters[i], kernel_size=kernel_sizes[i], padding=kernel_sizes[i]//2))
            self.bn_layers.append(nn.BatchNorm1d(cnn_filters[i]))
            self.act_layers.append(nn.ReLU())
            self.pool_layers.append(nn.MaxPool1d(pool_sizes[i]))
            self.dropout_layers.append(nn.Dropout(dropout * (0.5 + i * 0.1)))

        self.gru = nn.GRU(
            input_size=cnn_filters[-1],
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
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x_reshaped = x.reshape(batch_size * seq_len, -1)
        x_normalized = self.input_bn(x_reshaped)
        x = x_normalized.reshape(batch_size, seq_len, -1)

        x = x.permute(0, 2, 1)

        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)
            x = self.act_layers[i](x)
            x = self.pool_layers[i](x)
            x = self.dropout_layers[i](x)

        x = x.permute(0, 2, 1)
        
        gru_out, _ = self.gru(x)
        attention_weights = self.attention(gru_out)
        context_vector = torch.sum(attention_weights * gru_out, dim=1)
        output = self.fc(context_vector)
        return output

# ==============================================================================
# Model: best_model_tcn.pth
# Source: KFall Dataset/tcn_fall_prediction.py
# ==============================================================================
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        conv_out = self.conv(x)
        return conv_out[:, :, :-self.padding] if self.padding != 0 else conv_out

class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, dropout=0.3):
        super(TCNResidualBlock, self).__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.batch_norm2(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out = out + residual
        out = self.relu2(out)
        out = self.dropout2(out)
        return out

class TCN(nn.Module):
    def __init__(self, input_size=9, hidden_channels=[32, 64, 128], kernel_size=7, dropout=0.3):
        super(TCN, self).__init__()
        self.input_bn = nn.BatchNorm1d(input_size)
        
        layers = []
        num_levels = len(hidden_channels)
        in_channels = input_size
        for i in range(num_levels):
            dilation = 2 ** i
            out_channels = hidden_channels[i]
            layers.append(
                TCNResidualBlock(
                    in_channels, out_channels, kernel_size, stride=1, 
                    dilation=dilation, dropout=dropout
                )
            )
            in_channels = out_channels
        
        self.tcn_network = nn.Sequential(*layers)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1]),
            nn.Tanh(),
            nn.Linear(hidden_channels[-1], 1),
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels[-1] // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x_reshaped = x.reshape(batch_size * seq_len, -1)
        x_normalized = self.input_bn(x_reshaped)
        x = x_normalized.reshape(batch_size, seq_len, -1)
        
        x = x.permute(0, 2, 1)
        tcn_out = self.tcn_network(x)
        tcn_out = tcn_out.permute(0, 2, 1)
        
        attention_weights = self.attention(tcn_out)
        context_vector = torch.sum(attention_weights * tcn_out, dim=1)
        output = self.fc(context_vector)
        return output

# ==============================================================================
# Model: best_model_cnn_lstm_unbalanced.pth
# Source: KFall Dataset/cnn_lstm_fall_prediction_unbalanced.py
# ==============================================================================
class UnbalancedCNNLSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, num_layers=2, dropout=0.3):
        super(UnbalancedCNNLSTM, self).__init__()
        cnn_filters = CNN_FILTERS
        kernel_sizes = CNN_KERNEL_SIZES
        pool_sizes = CNN_POOL_SIZES
        
        self.input_bn = nn.BatchNorm1d(input_size)
        
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.act_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        self.conv_layers.append(nn.Conv1d(input_size, cnn_filters[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2))
        self.bn_layers.append(nn.BatchNorm1d(cnn_filters[0]))
        self.act_layers.append(nn.ReLU())
        self.pool_layers.append(nn.MaxPool1d(pool_sizes[0]))
        self.dropout_layers.append(nn.Dropout(dropout * 0.5))
        
        for i in range(1, len(cnn_filters)):
            self.conv_layers.append(nn.Conv1d(cnn_filters[i-1], cnn_filters[i], kernel_size=kernel_sizes[i], padding=kernel_sizes[i]//2))
            self.bn_layers.append(nn.BatchNorm1d(cnn_filters[i]))
            self.act_layers.append(nn.ReLU())
            self.pool_layers.append(nn.MaxPool1d(pool_sizes[i]))
            self.dropout_layers.append(nn.Dropout(dropout * (0.5 + i * 0.1)))
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # 全连接层 - 修改为匹配未采样模型的结构
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),  # 这里从128降到64
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 包含sigmoid以匹配训练好的模型
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x_reshaped = x.reshape(batch_size * seq_len, -1)
        x_normalized = self.input_bn(x_reshaped)
        x = x_normalized.reshape(batch_size, seq_len, -1)
        
        x = x.permute(0, 2, 1)
        
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

# ==============================================================================
# MODEL REGISTRY
# ==============================================================================
MODEL_REGISTRY = {
    'best_model.pth': OriginalLSTM,
    'best_model_lstm_regularized.pth': RegularizedLSTM,
    'best_model_gru.pth': StandardGRU,
    'best_model_bidirectional_gru.pth': BiGRU,
    'best_model_cnn_lstm.pth': CNNLSTM,
    'best_model_cnn_gru.pth': CNNGRU,
    'best_model_tcn.pth': TCN,
    'best_model_gru_unbalanced.pth': StandardGRU,
    'best_model_cnn_lstm_unbalanced.pth': UnbalancedCNNLSTM
}

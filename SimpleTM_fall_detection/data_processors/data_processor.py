import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class SensorDataProcessor:
    """用于处理传感器数据的类，为SimpleTM模型准备输入"""
    
    def __init__(self, window_size=128, stride=64, normalize=True):
        """
        初始化数据处理器
        
        参数:
            window_size (int): 滑动窗口大小（时间步）
            stride (int): 窗口滑动步长
            normalize (bool): 是否对数据进行标准化
        """
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.scaler = StandardScaler()
    
    def load_data(self, data_path, label_path=None):
        """
        加载传感器数据和标签
        
        参数:
            data_path (str): 传感器数据文件路径
            label_path (str): 标签文件路径（如果有）
            
        返回:
            pd.DataFrame: 处理后的数据框
        """
        # 加载数据
        data = pd.read_csv(data_path)
        
        # 如果提供了标签文件，则合并标签
        if label_path and os.path.exists(label_path):
            labels = pd.read_csv(label_path)
            # 假设数据和标签有相同的索引或时间戳
            data = pd.merge(data, labels, on='timestamp', how='left')
        
        return data
    
    def preprocess(self, data):
        """
        预处理数据：清洗、填充缺失值、标准化
        
        参数:
            data (pd.DataFrame): 原始数据框
            
        返回:
            pd.DataFrame: 预处理后的数据框
        """
        # 删除重复行
        data = data.drop_duplicates()
        
        # 填充缺失值
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # 仅选择传感器数据列（排除时间戳、标签、ID等非数值列）
        excluded_columns = [
            'timestamp', 'TimeStamp(s)', 'FrameCounter', 
            'label', 'activity', 'subject', 
            'participant_id', 'task_id', 'trial_id',
            'fall_description', 'fall_onset_frame', 'fall_impact_frame'
        ]
        
        # 确定真正的数值列
        sensor_columns = []
        for col in data.columns:
            if col not in excluded_columns:
                # 检查列是否全部为数值
                try:
                    pd.to_numeric(data[col])
                    sensor_columns.append(col)
                except:
                    print(f"警告: 列 '{col}' 包含非数值数据，将被排除在标准化之外")
        
        # 标准化
        if self.normalize and sensor_columns:
            # 先转换为numpy数组再标准化，避免pandas的索引问题
            data_values = data[sensor_columns].values
            data_scaled = self.scaler.fit_transform(data_values)
            data[sensor_columns] = data_scaled
        
        return data
    
    def create_windows(self, data, has_labels=True):
        """
        创建滑动窗口数据
        
        参数:
            data (pd.DataFrame): 预处理后的数据
            has_labels (bool): 数据是否包含标签
            
        返回:
            tuple: (X, y) 窗口化的特征和标签（如果有）
        """
        # 排除非特征列
        excluded_columns = [
            'timestamp', 'TimeStamp(s)', 'FrameCounter', 
            'label', 'activity', 'subject', 
            'participant_id', 'task_id', 'trial_id',
            'fall_description', 'fall_onset_frame', 'fall_impact_frame'
        ]
        
        sensor_columns = [col for col in data.columns 
                         if col not in excluded_columns]
        
        X_windows = []
        y_windows = []
        
        for i in range(0, len(data) - self.window_size + 1, self.stride):
            window = data[sensor_columns].iloc[i:i+self.window_size].values
            X_windows.append(window)
            
            if has_labels:
                # 针对跌倒检测，我们采用多数投票确定窗口的标签
                # 如果窗口内有任何跌倒样本，则将整个窗口标记为跌倒
                window_labels = data['label'].iloc[i:i+self.window_size].values
                # 假设1表示跌倒，0表示正常活动
                is_fall = 1 if 1 in window_labels else 0
                y_windows.append(is_fall)
        
        X = np.array(X_windows)
        
        if has_labels:
            y = np.array(y_windows)
            return X, y
        else:
            return X, None
    
    def process_for_simpleTM(self, X):
        """
        将窗口化数据转换为SimpleTM模型所需的格式
        
        参数:
            X (np.ndarray): 窗口化的特征数据，形状为(n_windows, window_size, n_features)
            
        返回:
            np.ndarray: 重塑后的数据，形状为(n_windows, n_features, window_size)
        """
        # SimpleTM需要的输入格式为(batch_size, n_features, seq_len)
        # 而我们的窗口化数据为(n_windows, window_size, n_features)
        # 因此需要转置最后两个维度
        return np.transpose(X, (0, 2, 1))

    def split_data(self, X, y=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """
        将数据拆分为训练集、验证集和测试集
        
        参数:
            X (np.ndarray): 特征数据
            y (np.ndarray): 标签数据
            train_ratio (float): 训练集比例
            val_ratio (float): 验证集比例
            test_ratio (float): 测试集比例
            random_state (int): 随机种子
            
        返回:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        np.random.seed(random_state)
        
        # 计算各集合的样本数
        n_samples = len(X)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        # 生成随机索引
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]
        
        # 分割数据
        X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
        
        if y is not None:
            y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            return X_train, X_val, X_test


class FallDetectionDataset(Dataset):
    """为跌倒检测任务创建PyTorch数据集"""
    
    def __init__(self, X, y=None):
        """
        初始化数据集
        
        参数:
            X (np.ndarray): 特征数据，形状为(n_samples, n_features, seq_len)
            y (np.ndarray): 标签数据，形状为(n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
    """
    创建PyTorch数据加载器
    
    参数:
        X_train, X_val, X_test (np.ndarray): 特征数据
        y_train, y_val, y_test (np.ndarray): 标签数据
        batch_size (int): 批次大小
        
    返回:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dataset = FallDetectionDataset(X_train, y_train)
    val_dataset = FallDetectionDataset(X_val, y_val)
    test_dataset = FallDetectionDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


def prepare_data_for_simpleTM(data_path, label_path=None, window_size=128, stride=64, 
                             train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                             batch_size=32, normalize=True):
    """
    准备SimpleTM模型所需的数据
    
    参数:
        data_path (str): 数据文件路径
        label_path (str): 标签文件路径（如果有）
        window_size (int): 滑动窗口大小
        stride (int): 窗口滑动步长
        train_ratio, val_ratio, test_ratio (float): 训练集、验证集、测试集比例
        batch_size (int): 批次大小
        normalize (bool): 是否标准化数据
        
    返回:
        tuple: (train_loader, val_loader, test_loader, scaler)
    """
    print(f"加载数据: {data_path}")
    
    # 初始化数据处理器
    processor = SensorDataProcessor(window_size=window_size, stride=stride, normalize=normalize)
    
    # 加载并预处理数据
    data = processor.load_data(data_path, label_path)
    print(f"原始数据形状: {data.shape}")
    
    # 检查列是否存在
    if 'label' not in data.columns:
        print("警告: 'label' 列不存在，尝试添加label列...")
        # 根据fall_onset_frame和fall_impact_frame添加标签
        if 'fall_onset_frame' in data.columns and 'fall_impact_frame' in data.columns:
            # 如果fall_onset_frame和fall_impact_frame都大于0，则标记为跌倒
            fall_mask = (data['fall_onset_frame'] > 0) & (data['fall_impact_frame'] > 0)
            
            # 初始化所有样本为非跌倒(0)
            data['label'] = 0
            
            # 对于每个满足条件的样本，处理跌倒区间
            for idx, row in data[fall_mask].iterrows():
                onset = int(row['fall_onset_frame'])
                impact = int(row['fall_impact_frame'])
                
                # 确保frame_counter列存在
                if 'FrameCounter' in data.columns:
                    # 从onset前到impact后的一定范围标记为跌倒(1)
                    data.loc[(data['FrameCounter'] >= onset) & (data['FrameCounter'] <= impact), 'label'] = 1
    
    data = processor.preprocess(data)
    print(f"预处理后数据形状: {data.shape}")
    
    # 创建窗口
    has_labels = 'label' in data.columns
    if has_labels:
        X, y = processor.create_windows(data, has_labels=True)
        print(f"正负样本分布: 正样本={sum(y)}, 负样本={len(y) - sum(y)}")
    else:
        X, y = processor.create_windows(data, has_labels=False)
    
    print(f"窗口化后数据形状: {X.shape}")
    
    # 转换为SimpleTM格式
    X = processor.process_for_simpleTM(X)
    print(f"转换后数据形状: {X.shape}")
    
    # 分割数据
    if has_labels:
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
            X, y, train_ratio, val_ratio, test_ratio)
        
        print(f"训练集: {X_train.shape}, {y_train.shape}")
        print(f"验证集: {X_val.shape}, {y_val.shape}")
        print(f"测试集: {X_test.shape}, {y_test.shape}")
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_dataloaders(
            X_train, X_val, X_test, y_train, y_val, y_test, batch_size)
        
        return train_loader, val_loader, test_loader, processor.scaler
    else:
        X_train, X_val, X_test = processor.split_data(X, None, train_ratio, val_ratio, test_ratio)
        
        print(f"训练集: {X_train.shape}")
        print(f"验证集: {X_val.shape}")
        print(f"测试集: {X_test.shape}")
        
        # 创建数据加载器（无标签）
        train_dataset = FallDetectionDataset(X_train)
        val_dataset = FallDetectionDataset(X_val)
        test_dataset = FallDetectionDataset(X_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader, processor.scaler 

def load_from_time_series_splits(data_dir, window_size=128, stride=64, batch_size=32, normalize=True):
    """
    从预分割的时间序列数据集加载数据
    
    参数:
        data_dir (str): 包含train.csv, val.csv, test.csv的目录路径
        window_size (int): 滑动窗口大小
        stride (int): 滑动窗口步长
        batch_size (int): 批次大小
        normalize (bool): 是否标准化数据
        
    返回:
        tuple: (train_loader, val_loader, test_loader, scaler)
    """
    print(f"从分割数据集加载数据: {data_dir}")
    
    # 创建数据处理器
    processor = SensorDataProcessor(window_size=window_size, stride=stride, normalize=normalize)
    
    # 加载预分割的数据集
    train_path = os.path.join(data_dir, 'train.csv')
    val_path = os.path.join(data_dir, 'val.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    # 加载训练集
    train_data = processor.load_data(train_path)
    train_data = processor.preprocess(train_data)
    
    # 加载验证集
    val_data = processor.load_data(val_path)
    val_data = processor.preprocess(val_data)
    
    # 加载测试集
    test_data = processor.load_data(test_path)
    test_data = processor.preprocess(test_data)
    
    # 检查标签列是否存在
    has_labels = 'label' in train_data.columns
    
    if has_labels:
        print("数据集包含标签列")
        # 为训练集创建窗口
        X_train, y_train = processor.create_windows(train_data, has_labels=True)
        X_train = processor.process_for_simpleTM(X_train)
        
        # 为验证集创建窗口
        X_val, y_val = processor.create_windows(val_data, has_labels=True)
        X_val = processor.process_for_simpleTM(X_val)
        
        # 为测试集创建窗口
        X_test, y_test = processor.create_windows(test_data, has_labels=True)
        X_test = processor.process_for_simpleTM(X_test)
        
        # 分类问题转换为整数类型
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        y_test = y_test.astype(int)
        
        # 打印正负样本分布
        print(f"训练集样本分布: 正样本={np.sum(y_train==1)}, 负样本={np.sum(y_train==0)}")
        print(f"验证集样本分布: 正样本={np.sum(y_val==1)}, 负样本={np.sum(y_val==0)}")
        print(f"测试集样本分布: 正样本={np.sum(y_test==1)}, 负样本={np.sum(y_test==0)}")
    else:
        print("数据集不包含标签列，假设为无监督任务")
        # 为训练集创建窗口
        X_train, _ = processor.create_windows(train_data, has_labels=False)
        X_train = processor.process_for_simpleTM(X_train)
        y_train = np.zeros(len(X_train))  # 创建假标签
        
        # 为验证集创建窗口
        X_val, _ = processor.create_windows(val_data, has_labels=False)
        X_val = processor.process_for_simpleTM(X_val)
        y_val = np.zeros(len(X_val))  # 创建假标签
        
        # 为测试集创建窗口
        X_test, _ = processor.create_windows(test_data, has_labels=False)
        X_test = processor.process_for_simpleTM(X_test)
        y_test = np.zeros(len(X_test))  # 创建假标签
    
    # 打印数据形状
    print(f"训练集: {X_train.shape}, {y_train.shape}")
    print(f"验证集: {X_val.shape}, {y_val.shape}")
    print(f"测试集: {X_test.shape}, {y_test.shape}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size
    )
    
    print(f"数据准备完成。训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}, 测试批次: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, processor.scaler 
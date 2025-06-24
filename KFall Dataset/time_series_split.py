import pandas as pd
import numpy as np
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """加载已处理的KFall数据集"""
    script_dir = Path(__file__).parent
    data_path = script_dir / 'processed_data' / 'kfall_processed_data.csv'
    
    print("正在加载数据...")
    df = pd.read_csv(data_path)
    
    print(f"数据形状: {df.shape}")
    print(f"标签分布:")
    print(df['label'].value_counts())
    
    return df

def create_time_series_splits(df, train_ratio=0.7, val_ratio=0.15):
    """
    创建保留时间序列的数据集划分
    
    这个函数通过按参与者和活动分组，并按时间顺序排列，保持时间序列的完整性
    每个参与者的数据内部按照实验编号的时间顺序进行划分
    
    Args:
        df: 完整的DataFrame
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        
    Returns:
        字典，包含训练集、验证集、测试集
    """
    print("\n创建时间序列数据划分...")
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # 按参与者分组
    unique_participants = df['participant_id'].unique()
    np.random.seed(42)  # 设置随机种子，确保结果可重现
    np.random.shuffle(unique_participants)  # 随机打乱参与者顺序
    
    # 将参与者分配到各个集合
    n_participants = len(unique_participants)
    n_train = int(n_participants * 0.7)
    n_val = int(n_participants * 0.15)
    
    train_participants = unique_participants[:n_train]
    val_participants = unique_participants[n_train:n_train+n_val]
    test_participants = unique_participants[n_train+n_val:]
    
    print(f"训练集参与者: {sorted(train_participants)} ({len(train_participants)}人)")
    print(f"验证集参与者: {sorted(val_participants)} ({len(val_participants)}人)")
    print(f"测试集参与者: {sorted(test_participants)} ({len(test_participants)}人)")
    
    # 创建数据集
    train_df = df[df['participant_id'].isin(train_participants)].copy()
    val_df = df[df['participant_id'].isin(val_participants)].copy()
    test_df = df[df['participant_id'].isin(test_participants)].copy()
    
    # 确保每个数据集内部按照时间序列排序
    for dataset in [train_df, val_df, test_df]:
        dataset.sort_values(by=['participant_id', 'task_id', 'trial_id', 'FrameCounter'], inplace=True)
    
    print(f"\n分割结果:")
    print(f"训练集: {len(train_df)} 行 ({len(train_df) / len(df):.1%})")
    print(f"验证集: {len(val_df)} 行 ({len(val_df) / len(df):.1%})")
    print(f"测试集: {len(test_df)} 行 ({len(test_df) / len(df):.1%})")
    
    # 检查标签分布
    print(f"\n训练集标签分布:")
    print(train_df['label'].value_counts())
    print(f"验证集标签分布:")
    print(val_df['label'].value_counts())
    print(f"测试集标签分布:")
    print(test_df['label'].value_counts())
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

def save_splits(splits, output_dir):
    """保存数据集划分"""
    print(f"\n保存时间序列数据集划分...")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 保存各个数据集
    for split_name, split_df in splits.items():
        output_file = output_path / f'{split_name}.csv'
        split_df.to_csv(output_file, index=False)
        print(f"已保存{split_name}数据到 {output_file} ({len(split_df)} 行)")

def main():
    """主函数"""
    print("=== KFall 数据集时间序列划分 ===\n")
    
    # 1. 加载数据
    df = load_data()
    
    # 2. 创建保留时间序列的数据集划分
    splits = create_time_series_splits(df)
    
    # 3. 保存数据集
    output_dir = 'time_series_splits'
    save_splits(splits, output_dir)
    
    print(f"\n=== 数据集时间序列划分完成 ===")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    main() 
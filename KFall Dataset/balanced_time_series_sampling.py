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
    print(f"跌倒帧比例: {df['label'].mean():.2%}")
    
    return df

def create_balanced_time_series_splits(df, train_ratio=0.7, val_ratio=0.15, target_ratio=1/3):
    """
    创建平衡的时间序列数据集划分，同时保持时间连续性
    
    这个函数通过以下步骤创建平衡数据集:
    1. 按参与者分组，保持每个参与者的数据完整性
    2. 对于每个参与者，保留所有跌倒序列，但对非跌倒帧进行采样
    3. 保持每个序列的时间连续性
    
    Args:
        df: 完整的DataFrame
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        target_ratio: 目标跌倒帧比例 (1:2 意味着跌倒帧占总帧数的1/3)
        
    Returns:
        字典，包含训练集、验证集、测试集
    """
    print("\n创建平衡的时间序列数据划分...")
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # 按参与者分组
    unique_participants = df['participant_id'].unique()
    np.random.seed(42)  # 设置随机种子，确保结果可重现
    np.random.shuffle(unique_participants)  # 随机打乱参与者顺序
    
    # 将参与者分配到各个集合
    n_participants = len(unique_participants)
    n_train = int(n_participants * train_ratio)
    n_val = int(n_participants * val_ratio)
    
    train_participants = unique_participants[:n_train]
    val_participants = unique_participants[n_train:n_train+n_val]
    test_participants = unique_participants[n_train+n_val:]
    
    print(f"训练集参与者: {sorted(train_participants)} ({len(train_participants)}人)")
    print(f"验证集参与者: {sorted(val_participants)} ({len(val_participants)}人)")
    print(f"测试集参与者: {sorted(test_participants)} ({len(test_participants)}人)")
    
    # 创建初始数据集划分
    train_df = df[df['participant_id'].isin(train_participants)].copy()
    val_df = df[df['participant_id'].isin(val_participants)].copy()
    test_df = df[df['participant_id'].isin(test_participants)].copy()
    
    # 对每个数据集应用平衡策略
    train_df_balanced = balance_dataset_with_sampling(train_df, target_ratio)
    val_df_balanced = balance_dataset_with_sampling(val_df, target_ratio)
    test_df_balanced = balance_dataset_with_sampling(test_df, target_ratio)
    
    # 确保每个数据集内部按照时间序列排序
    for dataset in [train_df_balanced, val_df_balanced, test_df_balanced]:
        dataset.sort_values(by=['participant_id', 'task_id', 'trial_id', 'FrameCounter'], inplace=True)
    
    print(f"\n分割结果:")
    print(f"训练集: {len(train_df_balanced)} 行 (原始: {len(train_df)} 行)")
    print(f"验证集: {len(val_df_balanced)} 行 (原始: {len(val_df)} 行)")
    print(f"测试集: {len(test_df_balanced)} 行 (原始: {len(test_df)} 行)")
    
    # 检查标签分布
    print(f"\n训练集标签分布:")
    print(train_df_balanced['label'].value_counts())
    print(f"跌倒帧比例: {train_df_balanced['label'].mean():.2%}")
    
    print(f"\n验证集标签分布:")
    print(val_df_balanced['label'].value_counts())
    print(f"跌倒帧比例: {val_df_balanced['label'].mean():.2%}")
    
    print(f"\n测试集标签分布:")
    print(test_df_balanced['label'].value_counts())
    print(f"跌倒帧比例: {test_df_balanced['label'].mean():.2%}")
    
    return {
        'train': train_df_balanced,
        'val': val_df_balanced,
        'test': test_df_balanced
    }

def balance_dataset_with_sampling(df, target_ratio=1/3, context_frames=50):
    """
    通过对非跌倒帧进行采样来平衡数据集，同时保持跌倒序列的上下文
    
    Args:
        df: 输入DataFrame
        target_ratio: 目标跌倒帧比例 (1:2 意味着跌倒帧占总帧数的1/3)
        context_frames: 跌倒前后需要保留的上下文帧数
        
    Returns:
        平衡后的DataFrame
    """
    # 按参与者、任务、试验分组
    groups = df.groupby(['participant_id', 'task_id', 'trial_id'])
    
    balanced_sequences = []
    
    # 遍历每个序列
    for name, group in groups:
        # 按时间顺序排序
        group = group.sort_values('FrameCounter')
        
        # 检查序列中是否包含跌倒帧
        fall_frames = group['label'].sum()
        
        if fall_frames > 0:
            # 包含跌倒帧，需要平衡
            balanced_seq = balance_sequence(group, target_ratio, context_frames)
            balanced_sequences.append(balanced_seq)
        else:
            # 不包含跌倒帧，选择性保留
            # 我们只保留一部分非跌倒序列，以维持整体平衡
            if np.random.random() < 0.2:  # 随机保留20%的非跌倒序列
                balanced_sequences.append(group)
    
    # 合并所有平衡后的序列
    if balanced_sequences:
        balanced_df = pd.concat(balanced_sequences, ignore_index=True)
        total_frames = len(balanced_df)
        fall_frames = balanced_df['label'].sum()
        fall_ratio = fall_frames / total_frames
        
        print(f"平衡后总帧数: {total_frames}, 跌倒帧: {fall_frames}, 跌倒比例: {fall_ratio:.2%}")
        return balanced_df
    else:
        print("警告: 没有找到可用序列，返回原始数据")
        return df

def balance_sequence(sequence, target_ratio=1/3, context_frames=50):
    """
    平衡单个序列，保留跌倒帧及其上下文，对其他非跌倒帧进行采样
    
    Args:
        sequence: 单个序列的DataFrame
        target_ratio: 目标跌倒帧比例
        context_frames: 跌倒前后需要保留的上下文帧数
        
    Returns:
        平衡后的序列
    """
    # 找出所有跌倒帧的索引
    fall_indices = sequence.index[sequence['label'] == 1].tolist()
    
    if not fall_indices:
        return sequence  # 如果没有跌倒帧，返回原始序列
    
    # 初始化保留帧的掩码
    keep_mask = np.zeros(len(sequence), dtype=bool)
    
    # 标记所有跌倒帧为保留
    keep_mask[sequence['label'] == 1] = True
    
    # 标记跌倒帧前后的上下文帧为保留
    for idx in fall_indices:
        # 相对于序列起始位置的索引
        rel_idx = idx - sequence.index[0]
        
        # 标记前context_frames帧为保留
        start_idx = max(0, rel_idx - context_frames)
        keep_mask[start_idx:rel_idx] = True
        
        # 标记后context_frames帧为保留
        end_idx = min(len(sequence), rel_idx + context_frames + 1)
        keep_mask[rel_idx+1:end_idx] = True
    
    # 计算当前保留的帧数
    fall_frames = sequence['label'].sum()
    context_frames_kept = keep_mask.sum() - fall_frames
    
    # 计算目标非跌倒帧数
    target_non_fall_frames = int(fall_frames * (1 - target_ratio) / target_ratio)
    
    # 如果保留的上下文帧已经超过目标非跌倒帧数，需要进行采样
    if context_frames_kept > target_non_fall_frames:
        # 找出所有被标记为保留的非跌倒帧
        context_indices = np.where(keep_mask & (sequence['label'].values == 0))[0]
        
        # 计算需要丢弃的帧数
        frames_to_drop = context_frames_kept - target_non_fall_frames
        
        # 如果需要丢弃的帧数大于0
        if frames_to_drop > 0:
            # 随机选择要丢弃的帧
            drop_indices = np.random.choice(context_indices, size=frames_to_drop, replace=False)
            keep_mask[drop_indices] = False
    else:
        # 如果保留的上下文帧不足，需要从其他非跌倒帧中采样
        # 找出所有未标记为保留的非跌倒帧
        non_context_indices = np.where((~keep_mask) & (sequence['label'].values == 0))[0]
        
        # 计算需要额外保留的帧数
        frames_to_keep = target_non_fall_frames - context_frames_kept
        
        # 如果有足够的非上下文帧可供选择
        if len(non_context_indices) > 0 and frames_to_keep > 0:
            # 随机选择要保留的帧
            keep_indices = np.random.choice(
                non_context_indices, 
                size=min(frames_to_keep, len(non_context_indices)), 
                replace=False
            )
            keep_mask[keep_indices] = True
    
    # 返回平衡后的序列
    balanced_sequence = sequence[keep_mask].copy()
    
    # 打印平衡前后的统计信息
    orig_fall_ratio = sequence['label'].mean()
    new_fall_ratio = balanced_sequence['label'].mean()
    
    print(f"序列 {sequence['participant_id'].iloc[0]}-{sequence['task_id'].iloc[0]}-{sequence['trial_id'].iloc[0]}: "
          f"原始 {len(sequence)} 帧 (跌倒比例: {orig_fall_ratio:.2%}), "
          f"平衡后 {len(balanced_sequence)} 帧 (跌倒比例: {new_fall_ratio:.2%})")
    
    return balanced_sequence

def save_splits(splits, output_dir):
    """保存数据集划分"""
    print(f"\n保存平衡的时间序列数据集划分...")
    
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
    print("=== KFall 数据集平衡时间序列划分 (跌倒:非跌倒 = 1:2) ===\n")
    
    # 1. 加载数据
    df = load_data()
    
    # 2. 创建保留时间序列的平衡数据集划分 (目标比例1:2，即跌倒帧占1/3)
    splits = create_balanced_time_series_splits(df, target_ratio=1/3)
    
    # 3. 保存数据集
    output_dir = 'balanced_time_series_splits_1to2'
    save_splits(splits, output_dir)
    
    print(f"\n=== 平衡数据集时间序列划分完成 ===")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    main() 
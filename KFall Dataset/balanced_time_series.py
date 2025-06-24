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
    2. 对于每个参与者，保留所有跌倒序列及其前后上下文
    3. 从非跌倒序列中采样，使得最终跌倒:非跌倒比例达到目标比例
    4. 保持每个序列的时间连续性
    
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
    train_df_balanced = balance_dataset_with_context(train_df, target_ratio)
    val_df_balanced = balance_dataset_with_context(val_df, target_ratio)
    test_df_balanced = balance_dataset_with_context(test_df, target_ratio)
    
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

def balance_dataset_with_context(df, target_ratio=1/3, context_frames=100):
    """
    平衡数据集，同时保持跌倒序列的上下文
    
    Args:
        df: 输入DataFrame
        target_ratio: 目标跌倒帧比例 (1:2 意味着跌倒帧占总帧数的1/3)
        context_frames: 跌倒前后需要保留的上下文帧数
        
    Returns:
        平衡后的DataFrame
    """
    # 按参与者、任务、试验分组
    groups = df.groupby(['participant_id', 'task_id', 'trial_id'])
    
    fall_sequences = []
    non_fall_sequences = []
    
    # 遍历每个序列
    for name, group in groups:
        # 按时间顺序排序
        group = group.sort_values('FrameCounter')
        
        # 检查序列中是否包含跌倒帧
        if group['label'].sum() > 0:
            # 包含跌倒帧，保留整个序列
            fall_sequences.append(group)
        else:
            # 不包含跌倒帧，加入候选池
            non_fall_sequences.append(group)
    
    # 合并所有跌倒序列
    if fall_sequences:
        fall_df = pd.concat(fall_sequences, ignore_index=True)
        fall_frames = len(fall_df)
        fall_count = fall_df['label'].sum()
    else:
        fall_df = pd.DataFrame(columns=df.columns)
        fall_frames = 0
        fall_count = 0
    
    print(f"跌倒序列: {len(fall_sequences)} 个, 共 {fall_frames} 帧, 其中跌倒帧 {fall_count} 帧")
    
    # 如果没有跌倒序列，直接返回原始数据
    if not fall_sequences:
        print("警告: 没有找到跌倒序列，返回原始数据")
        return df
    
    # 计算需要的非跌倒帧数量，以达到目标比例
    # 目标比例: fall_count / (fall_count + non_fall_needed) = target_ratio
    # 因此: non_fall_needed = fall_count * (1 - target_ratio) / target_ratio
    non_fall_frames_needed = int(fall_count * (1 - target_ratio) / target_ratio)
    
    print(f"目标跌倒比例: {target_ratio:.2%} (跌倒:非跌倒 = 1:2)")
    print(f"跌倒帧数量: {fall_count}")
    print(f"需要的非跌倒帧数量: {non_fall_frames_needed}")
    
    # 当前跌倒序列中的非跌倒帧
    existing_non_fall_frames = fall_frames - fall_count
    print(f"跌倒序列中的非跌倒帧: {existing_non_fall_frames}")
    
    # 如果跌倒序列中的非跌倒帧已经超过所需，需要减少
    if existing_non_fall_frames > non_fall_frames_needed:
        print("跌倒序列中的非跌倒帧已超过目标比例，需要减少非跌倒帧")
        # 我们不能简单地删除帧，因为这会破坏时间连续性
        # 所以我们保持所有跌倒序列完整，并接受稍微偏离目标比例
        print(f"保留所有跌倒序列完整，实际非跌倒帧: {existing_non_fall_frames}")
        return fall_df
    
    # 如果跌倒序列中的非跌倒帧不足，需要从非跌倒序列中添加
    additional_non_fall_needed = non_fall_frames_needed - existing_non_fall_frames
    print(f"还需要额外的非跌倒帧: {additional_non_fall_needed}")
    
    # 随机选择非跌倒序列，直到达到目标帧数
    np.random.shuffle(non_fall_sequences)
    selected_non_fall_sequences = []
    current_non_fall_frames = 0
    
    for seq in non_fall_sequences:
        if current_non_fall_frames >= additional_non_fall_needed:
            break
        
        selected_non_fall_sequences.append(seq)
        current_non_fall_frames += len(seq)
    
    print(f"选择的非跌倒序列: {len(selected_non_fall_sequences)} 个, 共 {current_non_fall_frames} 帧")
    
    # 合并跌倒序列和选择的非跌倒序列
    balanced_df = pd.concat(fall_sequences + selected_non_fall_sequences, ignore_index=True)
    
    # 检查最终的跌倒比例
    final_fall_ratio = balanced_df['label'].sum() / len(balanced_df)
    print(f"最终跌倒帧比例: {final_fall_ratio:.2%}")
    
    return balanced_df

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
    output_dir = 'balanced_time_series_splits'
    save_splits(splits, output_dir)
    
    print(f"\n=== 平衡数据集时间序列划分完成 ===")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    main() 
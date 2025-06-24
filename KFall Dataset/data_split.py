import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_reclassified_data():
    """加载重新分类后的数据"""
    script_dir = Path(__file__).parent
    data_path = script_dir / 'reclassified_data' / 'kfall_reclassified_data.csv'
    
    print("正在加载重新分类后的数据...")
    df = pd.read_csv(data_path)
    
    print(f"数据形状: {df.shape}")
    print(f"活动类型分布:")
    print(df['activity_type'].value_counts())
    print(f"\n标签分布:")
    print(df['label'].value_counts())
    
    return df

def create_sliding_windows(df, window_size=50, stride=25):
    """创建滑动窗口时间序列数据"""
    print(f"\n创建滑动窗口 (窗口大小: {window_size}, 步长: {stride})...")
    
    # 所有需要保留的列名
    all_cols = df.columns.tolist()
    
    window_data_list = []
    window_id_counter = 0 # 用于给每个窗口一个唯一ID
    
    # 按参与者、任务、试验分组处理
    for (participant, task, trial), group in df.groupby(['participant_id', 'task_id', 'trial_id']):
        group_labels = group['label'].values
        
        # 创建滑动窗口
        for i in range(0, len(group) - window_size + 1, stride):
            window_df = group.iloc[i:i + window_size].copy() # 获取原始DataFrame切片
            
            # 窗口标签：如果窗口内任何一帧是跌倒，则整个窗口标记为跌倒
            window_label = 1 if np.any(group_labels[i:i + window_size]) else 0
            
            # 添加窗口ID到DataFrame
            window_df['window_id'] = window_id_counter
            window_df['window_label'] = window_label # 将窗口标签也作为列保存
            
            window_data_list.append({
                'window_id': window_id_counter,
                'frames_data': window_df, # 存储包含所有原始列的DataFrame
                'window_label': window_label,
                'participant_id': participant,
                'task_id': task,
                'trial_id': trial,
                'activity_type': group['activity_type'].iloc[0],
                'fall_description': group['fall_description'].iloc[0]
            })
            window_id_counter += 1
    
    print(f"共创建 {len(window_data_list)} 个滑动窗口。")
    # 打印窗口标签分布
    labels = [w['window_label'] for w in window_data_list]
    print(f"窗口标签分布:")
    print(pd.Series(labels).value_counts())
    
    # 打印活动类型分布
    activity_types = [w['activity_type'] for w in window_data_list]
    print(f"活动类型分布:")
    print(pd.Series(activity_types).value_counts())
    
    return window_data_list

def balance_samples(window_data_list, ratio=2):
    """平衡正负样本比例"""
    print(f"\n平衡正负样本比例 (非跌倒:跌倒 = {ratio}:1)...")
    
    fall_windows = [w for w in window_data_list if w['window_label'] == 1]
    non_fall_windows = [w for w in window_data_list if w['window_label'] == 0]
    
    print(f"原始数据:")
    print(f"  跌倒样本: {len(fall_windows)}")
    print(f"  非跌倒样本: {len(non_fall_windows)}")
    if len(fall_windows) > 0:
        print(f"  比例: {len(non_fall_windows)/len(fall_windows):.2f}:1")
    else:
        print(f"  无跌倒样本。")
    
    # 计算需要的非跌倒样本数量
    target_non_fall_count = len(fall_windows) * ratio
    
    if len(non_fall_windows) > target_non_fall_count:
        # 随机选择指定数量的非跌倒样本
        selected_non_fall_windows = np.random.choice(
            non_fall_windows, 
            size=int(target_non_fall_count), 
            replace=False
        ).tolist()
        print(f"  随机选择 {int(target_non_fall_count)} 个非跌倒样本")
    else:
        # 如果非跌倒样本不够，使用所有样本
        selected_non_fall_windows = non_fall_windows
        print(f"  使用所有 {len(non_fall_windows)} 个非跌倒样本")
    
    # 合并样本
    balanced_window_data_list = fall_windows + selected_non_fall_windows
    np.random.shuffle(balanced_window_data_list) # 打乱顺序
    
    print(f"平衡后数据:")
    print(f"  总样本数: {len(balanced_window_data_list)}")
    balanced_fall_count = sum(1 for w in balanced_window_data_list if w['window_label'] == 1)
    balanced_non_fall_count = sum(1 for w in balanced_window_data_list if w['window_label'] == 0)
    print(f"  跌倒样本: {balanced_fall_count}")
    print(f"  非跌倒样本: {balanced_non_fall_count}")
    if balanced_fall_count > 0:
        print(f"  比例: {balanced_non_fall_count/balanced_fall_count:.2f}:1")
    else:
        print(f"  无跌倒样本。")
    
    return balanced_window_data_list

def split_data_by_activity(window_data_list):
    """按活动类型分别划分数据"""
    print(f"\n按活动类型划分数据...")
    
    sitting_data = [w for w in window_data_list if w['activity_type'] == 'sitting']
    walking_data = [w for w in window_data_list if w['activity_type'] == 'walking']
    
    print(f"坐姿数据: {len(sitting_data)} 样本")
    print(f"走路数据: {len(walking_data)} 样本")
    
    return sitting_data, walking_data

def create_train_val_test_splits(window_data_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """创建训练集、验证集、测试集"""
    print(f"\n创建数据集划分 (训练:{train_ratio:.0%}, 验证:{val_ratio:.0%}, 测试:{test_ratio:.0%})...")
    
    # 按参与者分组，确保同一参与者的数据不会跨集
    unique_participants = list(set([w['participant_id'] for w in window_data_list]))
    np.random.shuffle(unique_participants)
    
    # 分配参与者到不同集合
    n_participants = len(unique_participants)
    n_train_participants = int(n_participants * train_ratio)
    n_val_participants = int(n_participants * val_ratio)
    
    train_participants = unique_participants[:n_train_participants]
    val_participants = unique_participants[n_train_participants:n_train_participants + n_val_participants]
    test_participants = unique_participants[n_train_participants + n_val_participants:]
    
    # 提取数据
    train_data = [w for w in window_data_list if w['participant_id'] in train_participants]
    val_data = [w for w in window_data_list if w['participant_id'] in val_participants]
    test_data = [w for w in window_data_list if w['participant_id'] in test_participants]
    
    print(f"训练集: {len(train_data)} 样本 ({len(train_participants)} 参与者)")
    print(f"验证集: {len(val_data)} 样本 ({len(val_participants)} 参与者)")
    print(f"测试集: {len(test_data)} 样本 ({len(test_participants)} 参与者)")
    
    # 检查标签分布
    for name, data in [('训练集', train_data), ('验证集', val_data), ('测试集', test_data)]:
        fall_count = sum(1 for w in data if w['window_label'] == 1)
        non_fall_count = sum(1 for w in data if w['window_label'] == 0)
        if fall_count > 0:
            print(f"{name}标签分布: 跌倒 {fall_count}, 非跌倒 {non_fall_count}, 比例 {non_fall_count/fall_count:.2f}:1")
        else:
            print(f"{name}标签分布: 无跌倒样本，非跌倒 {non_fall_count}")
    
    # 返回字典格式
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    return splits

def save_split_data(sitting_splits, walking_splits, output_dir):
    """保存划分后的数据"""
    print(f"\n保存划分后的数据...")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 保存坐姿数据
    sitting_dir = output_dir / 'sitting'
    sitting_dir.mkdir(exist_ok=True)
    
    for split_name, split_data_list in sitting_splits.items():
        if len(split_data_list) > 0:
            # 将所有窗口的frames_data拼接起来
            concatenated_df = pd.concat([w['frames_data'] for w in split_data_list], ignore_index=True)
            
            # 删除不需要的列
            if 'window_id' in concatenated_df.columns:
                concatenated_df = concatenated_df.drop(columns=['window_id'])
            if 'window_label' in concatenated_df.columns:
                concatenated_df = concatenated_df.drop(columns=['window_label'])
            
            output_file = sitting_dir / f'{split_name}.csv'
            concatenated_df.to_csv(output_file, index=False)
            print(f"已保存坐姿{split_name}数据到 {output_file} ({len(concatenated_df)} 行)")
        else:
            print(f"坐姿{split_name}数据为空，跳过保存。")
            
    # 保存走路数据
    walking_dir = output_dir / 'walking'
    walking_dir.mkdir(exist_ok=True)
    
    for split_name, split_data_list in walking_splits.items():
        if len(split_data_list) > 0:
            # 将所有窗口的frames_data拼接起来
            concatenated_df = pd.concat([w['frames_data'] for w in split_data_list], ignore_index=True)
            
            # 删除不需要的列
            if 'window_id' in concatenated_df.columns:
                concatenated_df = concatenated_df.drop(columns=['window_id'])
            if 'window_label' in concatenated_df.columns:
                concatenated_df = concatenated_df.drop(columns=['window_label'])
            
            output_file = walking_dir / f'{split_name}.csv'
            concatenated_df.to_csv(output_file, index=False)
            print(f"已保存走路{split_name}数据到 {output_file} ({len(concatenated_df)} 行)")
        else:
            print(f"走路{split_name}数据为空，跳过保存。")

def main():
    """主函数"""
    print("=== KFall 数据集划分 ===\n")
    
    # 设置随机种子
    np.random.seed(42)
    
    # 1. 加载数据
    df = load_reclassified_data()
    
    # 2. 创建滑动窗口
    window_data_list = create_sliding_windows(df)
    
    # 3. 平衡样本 (非跌倒:跌倒 = 2:1)
    balanced_window_data_list = balance_samples(window_data_list, ratio=2)
    
    # 4. 按活动类型划分数据
    sitting_data, walking_data = split_data_by_activity(balanced_window_data_list)
    
    # 5. 创建训练集、验证集、测试集
    sitting_splits = create_train_val_test_splits(sitting_data)
    walking_splits = create_train_val_test_splits(walking_data)
    
    # 6. 保存划分后的数据
    output_dir = Path('split_data')
    save_split_data(sitting_splits, walking_splits, output_dir)
    
    print(f"\n=== 数据划分完成 ===")

if __name__ == "__main__":
    main() 
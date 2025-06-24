import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_fall_descriptions():
    """分析fall_description的分布"""
    script_dir = Path(__file__).parent
    data_path = script_dir / 'processed_data' / 'kfall_processed_data.csv'
    
    print("正在加载数据...")
    df = pd.read_csv(data_path)
    
    print(f"数据形状: {df.shape}")
    print(f"\nfall_description唯一值:")
    print(df['fall_description'].unique())
    print(f"\nfall_description分布:")
    print(df['fall_description'].value_counts())
    
    return df

def classify_sitting_walking(df):
    """将数据分为坐和走两大类"""
    print("\n开始重新分类数据...")
    
    # 定义分类规则
    sitting_keywords = ['sit', 'sitting', 'chair', 'bed', 'sofa', 'seat', 'get up']
    walking_keywords = ['walk', 'walking', 'stand', 'standing', 'move', 'movement']
    
    def classify_activity(description):
        """根据描述分类活动"""
        if pd.isna(description):
            return 'unknown'
        
        description_lower = str(description).lower()
        
        # 检查是否包含坐相关关键词（包括get up）
        for keyword in sitting_keywords:
            if keyword in description_lower:
                return 'sitting'
        
        # 检查是否包含走相关关键词
        for keyword in walking_keywords:
            if keyword in description_lower:
                return 'walking'
        
        # 如果都不匹配，根据具体描述判断
        if 'fall' in description_lower:
            # 跌倒事件，需要进一步判断
            if any(keyword in description_lower for keyword in sitting_keywords):
                return 'sitting'
            elif any(keyword in description_lower for keyword in walking_keywords):
                return 'walking'
            else:
                return 'walking'  # 默认跌倒为走路时发生
        else:
            return 'unknown'
    
    # 添加新的分类列
    df['activity_type'] = df['fall_description'].apply(classify_activity)
    
    print(f"\n分类结果:")
    print(df['activity_type'].value_counts())
    
    # 显示每个分类的详细描述
    print(f"\n坐姿活动详细描述:")
    sitting_descriptions = df[df['activity_type'] == 'sitting']['fall_description'].unique()
    for desc in sitting_descriptions:
        print(f"  - {desc}")
    
    print(f"\n走路活动详细描述:")
    walking_descriptions = df[df['activity_type'] == 'walking']['fall_description'].unique()
    for desc in walking_descriptions:
        print(f"  - {desc}")
    
    print(f"\n未知活动详细描述:")
    unknown_descriptions = df[df['activity_type'] == 'unknown']['fall_description'].unique()
    for desc in unknown_descriptions:
        print(f"  - {desc}")
    
    return df

def save_reclassified_data(df):
    """保存重新分类的数据"""
    script_dir = Path(__file__).parent
    
    # 创建新的文件夹
    reclassified_dir = script_dir / 'reclassified_data'
    reclassified_dir.mkdir(exist_ok=True)
    
    # 分别保存坐姿和走路数据
    sitting_data = df[df['activity_type'] == 'sitting'].copy()
    walking_data = df[df['activity_type'] == 'walking'].copy()
    
    # 保存完整数据（包含新分类）
    df.to_csv(str(reclassified_dir / 'kfall_reclassified_data.csv'), index=False)
    
    # 保存分类后的数据
    sitting_data.to_csv(str(reclassified_dir / 'sitting_activities.csv'), index=False)
    walking_data.to_csv(str(reclassified_dir / 'walking_activities.csv'), index=False)
    
    # 保存数据信息
    info_data = {
        'total_records': len(df),
        'sitting_records': len(sitting_data),
        'walking_records': len(walking_data),
        'unknown_records': len(df[df['activity_type'] == 'unknown']),
        'sitting_percentage': len(sitting_data) / len(df) * 100,
        'walking_percentage': len(walking_data) / len(df) * 100,
        'unknown_percentage': len(df[df['activity_type'] == 'unknown']) / len(df) * 100
    }
    
    info_df = pd.DataFrame([info_data])
    info_df.to_csv(str(reclassified_dir / 'reclassification_info.csv'), index=False)
    
    print(f"\n数据已保存到 {reclassified_dir}")
    print(f"总记录数: {info_data['total_records']:,}")
    print(f"坐姿活动: {info_data['sitting_records']:,} ({info_data['sitting_percentage']:.1f}%)")
    print(f"走路活动: {info_data['walking_records']:,} ({info_data['walking_percentage']:.1f}%)")
    print(f"未知活动: {info_data['unknown_records']:,} ({info_data['unknown_percentage']:.1f}%)")
    
    return reclassified_dir

def main():
    """主函数"""
    print("=== KFall 数据重新分类 ===\n")
    
    # 1. 分析原始数据
    df = analyze_fall_descriptions()
    
    # 2. 重新分类
    df_reclassified = classify_sitting_walking(df)
    
    # 3. 保存数据
    output_dir = save_reclassified_data(df_reclassified)
    
    print(f"\n=== 重新分类完成 ===")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    main() 
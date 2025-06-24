import pandas as pd
import os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class KFallDataPreprocessor:
    def __init__(self, dataset_path):
        """
        初始化KFall数据预处理器
        
        Args:
            dataset_path: KFall数据集根目录路径
        """
        self.dataset_path = Path(dataset_path)
        self.label_data_path = self.dataset_path / 'label_data'
        self.sensor_data_path = self.dataset_path / 'sensor_data'
        
        # 存储处理后的数据
        self.processed_data = []
        self.data_info = []
        
    def parse_task_id(self, task_code):
        """
        解析Task Code (Task ID)列，提取括号内的数字
        
        Args:
            task_code: 如 "F01 (20)"
            
        Returns:
            int: 括号内的数字，如 20
        """
        if pd.isna(task_code):
            return None
        
        # 查找括号内的数字
        import re
        match = re.search(r'\((\d+)\)', str(task_code))
        if match:
            return int(match.group(1))
        return None
    
    def parse_trial_id(self, trial_id):
        """
        解析Trial ID，转换为R格式
        
        Args:
            trial_id: 如 1, 2, 3...
            
        Returns:
            str: R格式，如 "R01", "R02", "R03"
        """
        if pd.isna(trial_id):
            return None
        
        return f"R{int(trial_id):02d}"
    
    def load_label_data(self, participant_id):
        """
        加载指定参与者的标签数据
        
        Args:
            participant_id: 参与者ID，如 "SA06"
            
        Returns:
            pd.DataFrame: 处理后的标签数据
        """
        label_file = self.label_data_path / f"{participant_id}_label.xlsx"
        
        if not label_file.exists():
            print(f"警告: 标签文件 {label_file} 不存在")
            return pd.DataFrame()
        
        # 读取Excel文件
        df = pd.read_excel(label_file)
        
        # 清理数据：移除完全为空的行
        df = df.dropna(how='all')
        
        # 解析Task ID和Trial ID
        df['Task_ID_parsed'] = df['Task Code (Task ID)'].apply(self.parse_task_id)
        df['Trial_ID_parsed'] = df['Trial ID'].apply(self.parse_trial_id)
        
        # 只保留有效的跌倒数据（有起止帧标记的）
        df = df.dropna(subset=['Fall_onset_frame', 'Fall_impact_frame', 'Task_ID_parsed', 'Trial_ID_parsed'])
        
        return df
    
    def load_sensor_data(self, participant_id, task_id, trial_id):
        """
        加载指定参与者的传感器数据
        
        Args:
            participant_id: 参与者ID，如 "SA06"
            task_id: 任务ID，如 20
            trial_id: 试验ID，如 "R01"
            
        Returns:
            pd.DataFrame: 传感器数据
        """
        # 构建文件名
        # 注意：文件名格式是 S06T20R01.csv，但参与者ID是SA06
        # 需要去掉SA中的A
        participant_short = participant_id.replace('SA', 'S')
        filename = f"{participant_short}T{int(task_id):02d}{trial_id}.csv"
        
        sensor_file = self.sensor_data_path / participant_id / filename
        
        if not sensor_file.exists():
            print(f"警告: 传感器文件 {sensor_file} 不存在")
            return pd.DataFrame()
        
        # 读取CSV文件
        df = pd.read_csv(sensor_file)
        
        return df
    
    def label_sensor_data(self, sensor_df, fall_ranges):
        """
        为传感器数据添加标签
        
        Args:
            sensor_df: 传感器数据DataFrame
            fall_ranges: 跌倒帧范围列表，每个元素为 (onset, impact)
            
        Returns:
            pd.DataFrame: 带标签的传感器数据
        """
        # 复制数据，避免修改原始数据
        df = sensor_df.copy()
        
        # 初始化标签列
        df['label'] = 0
        
        # 为每个跌倒范围标记标签
        for onset, impact in fall_ranges:
            # 标记跌倒帧（包括起止帧）
            mask = (df['FrameCounter'] >= onset) & (df['FrameCounter'] <= impact)
            df.loc[mask, 'label'] = 1
        
        return df
    
    def process_participant(self, participant_id):
        """
        处理单个参与者的所有数据
        
        Args:
            participant_id: 参与者ID，如 "SA06"
        """
        print(f"正在处理参与者: {participant_id}")
        
        # 加载标签数据
        label_df = self.load_label_data(participant_id)
        
        if label_df.empty:
            print(f"参与者 {participant_id} 没有有效的标签数据")
            return
        
        # 按任务和试验分组处理
        for _, row in label_df.iterrows():
            task_id = row['Task_ID_parsed']
            trial_id = row['Trial_ID_parsed']
            onset_frame = row['Fall_onset_frame']
            impact_frame = row['Fall_impact_frame']
            description = row['Description']
            
            # 加载传感器数据
            sensor_df = self.load_sensor_data(participant_id, task_id, trial_id)
            
            if sensor_df.empty:
                continue
            
            # 添加标签
            labeled_df = self.label_sensor_data(sensor_df, [(onset_frame, impact_frame)])
            
            # 添加元数据列
            labeled_df['participant_id'] = participant_id
            labeled_df['task_id'] = task_id
            labeled_df['trial_id'] = trial_id
            labeled_df['fall_description'] = description
            labeled_df['fall_onset_frame'] = onset_frame
            labeled_df['fall_impact_frame'] = impact_frame
            
            # 存储处理后的数据
            self.processed_data.append(labeled_df)
            
            # 记录数据信息
            self.data_info.append({
                'participant_id': participant_id,
                'task_id': task_id,
                'trial_id': trial_id,
                'description': description,
                'total_frames': len(labeled_df),
                'fall_frames': labeled_df['label'].sum(),
                'non_fall_frames': len(labeled_df) - labeled_df['label'].sum(),
                'fall_ratio': labeled_df['label'].mean()
            })
            
            print(f"  处理完成: {participant_id} T{int(task_id):02d}{trial_id} - "
                  f"总帧数: {len(labeled_df)}, 跌倒帧: {labeled_df['label'].sum()}")
    
    def process_all_participants(self):
        """
        处理所有参与者的数据
        """
        print("开始处理KFall数据集...")
        
        # 获取所有参与者ID
        label_files = list(self.label_data_path.glob('SA*_label.xlsx'))
        participant_ids = [f.stem.replace('_label', '') for f in label_files]
        
        print(f"找到 {len(participant_ids)} 个参与者")
        print(f"参与者ID列表: {participant_ids[:5]}...")  # 显示前5个
        
        # 处理每个参与者
        for participant_id in sorted(participant_ids):
            self.process_participant(participant_id)
        
        print("所有参与者数据处理完成！")
    
    def save_processed_data(self, output_dir='processed_data'):
        """
        保存处理后的数据
        
        Args:
            output_dir: 输出目录
        """
        if not self.processed_data:
            print("没有处理后的数据可保存")
            return None, None
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 合并所有数据
        combined_df = pd.concat(self.processed_data, ignore_index=True)
        
        # 保存完整数据
        combined_file = output_path / 'kfall_processed_data.csv'
        combined_df.to_csv(combined_file, index=False)
        print(f"完整数据已保存到: {combined_file}")
        print(f"数据形状: {combined_df.shape}")
        
        # 保存数据信息摘要
        info_df = pd.DataFrame(self.data_info)
        info_file = output_path / 'kfall_data_info.csv'
        info_df.to_csv(info_file, index=False)
        print(f"数据信息已保存到: {info_file}")
        
        # 打印统计信息
        self.print_statistics(combined_df, info_df)
        
        return combined_df, info_df
    
    def print_statistics(self, combined_df, info_df):
        """
        打印数据统计信息
        
        Args:
            combined_df: 合并后的数据
            info_df: 数据信息
        """
        print("\n=== 数据统计信息 ===")
        print(f"总参与者数: {combined_df['participant_id'].nunique()}")
        print(f"总任务数: {combined_df['task_id'].nunique()}")
        print(f"总试验数: {len(info_df)}")
        print(f"总帧数: {len(combined_df):,}")
        print(f"跌倒帧数: {combined_df['label'].sum():,}")
        print(f"非跌倒帧数: {len(combined_df) - combined_df['label'].sum():,}")
        print(f"跌倒比例: {combined_df['label'].mean():.2%}")
        
        print("\n=== 参与者统计 ===")
        participant_stats = combined_df.groupby('participant_id').agg({
            'label': ['count', 'sum', 'mean']
        }).round(4)
        participant_stats.columns = ['总帧数', '跌倒帧数', '跌倒比例']
        print(participant_stats.head(10))
        
        print("\n=== 跌倒类型统计 ===")
        fall_type_stats = combined_df.groupby('fall_description').agg({
            'label': ['count', 'sum', 'mean']
        }).round(4)
        fall_type_stats.columns = ['总帧数', '跌倒帧数', '跌倒比例']
        print(fall_type_stats)

def main():
    """
    主函数
    """
    # 设置数据集路径 - 使用当前目录
    dataset_path = "."
    
    # 创建预处理器
    preprocessor = KFallDataPreprocessor(dataset_path)
    
    # 处理所有数据
    preprocessor.process_all_participants()
    
    # 保存处理后的数据
    combined_df, info_df = preprocessor.save_processed_data()
    
    if combined_df is not None:
        print("\n数据预处理完成！")
        print("生成的文件:")
        print("1. processed_data/kfall_processed_data.csv - 完整的带标签数据")
        print("2. processed_data/kfall_data_info.csv - 数据信息摘要")
    else:
        print("\n数据预处理失败，请检查数据集路径和文件结构")

if __name__ == "__main__":
    main() 
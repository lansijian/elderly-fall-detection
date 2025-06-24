import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os
import textwrap
from matplotlib.font_manager import FontProperties
warnings.filterwarnings('ignore')

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 字体配置
def setup_chinese_font():
    """设置中文字体 - 使用本地字体文件"""
    script_dir = Path(__file__).parent
    
    # 尝试多个可能的字体文件
    font_files = [
        'SimHei.ttf',
        'simhei.ttf', 
        'Microsoft YaHei.ttf',
        'msyh.ttf',
        'SimSun.ttf',
        'simsun.ttf'
    ]
    
    for font_file in font_files:
        font_path = script_dir / font_file
        if font_path.exists():
            try:
                # 测试字体是否可用
                test_font = FontProperties(fname=str(font_path))
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, '测试中文', fontproperties=test_font, fontsize=12)
                plt.close(fig)
                
                print(f"成功加载中文字体: {font_file}")
                return test_font
            except Exception as e:
                print(f"字体 {font_file} 测试失败: {e}")
                continue
    
    print("警告: 未找到可用的中文字体文件，将使用英文标签")
    return None

# 设置字体
CHINESE_FONT = setup_chinese_font()

# 设置图表样式
plt.style.use('default')
sns.set_palette("husl")

# 标签换行函数
def wrap_labels(labels, width=15):
    """对长标签进行换行处理"""
    wrapped_labels = []
    for label in labels:
        if len(str(label)) > width:
            wrapped = '\n'.join(textwrap.wrap(str(label), width=width))
            wrapped_labels.append(wrapped)
        else:
            wrapped_labels.append(str(label))
    return wrapped_labels

# 定义标签（全部使用英文）
LABELS = {
    'data_overview': 'Data Overview',         # 数据概览
    'non_fall': 'Non-Fall',                  # 非跌倒
    'fall': 'Fall',                          # 跌倒
    'total_data': 'Total Data',              # 总数据量
    'participants': 'Participants',          # 参与者数
    'trials': 'Trials',                      # 试验数
    'participant_distribution': 'Participant Distribution', # 各参与者数据分布
    'participant_id': 'Participant ID',      # 参与者ID
    'frame_count': 'Frame Count',            # 帧数
    'total_frames': 'Total Frames',          # 总帧数
    'fall_frames': 'Fall Frames',            # 跌倒帧数
    'fall_type_distribution': 'Fall Type Distribution', # 跌倒类型分布
    'fall_frames_count': 'Fall Frames Count',# 跌倒帧数
    'sensor_distribution': 'Sensor Distribution', # 传感器数据分布
    'sensor_type': 'Sensor Type',            # 传感器类型
    'value': 'Value',                        # 数值
    'time_series_example': 'Time Series Example', # 时间序列示例
    'frame_number': 'Frame Number',          # 帧数
    'acceleration': 'Acceleration',          # 加速度值
    'fall_region': 'Fall Region',            # 跌倒区域
    'fall_detection_stats': 'Fall Detection Statistics', # 试验跌倒检测统计
    'total_frames_x': 'Total Frames',        # 总帧数
    'fall_frames_y': 'Fall Frames',          # 跌倒帧数
    'fall_ratio': 'Fall Ratio',              # 跌倒比例
    'participant_activity_stats': 'Participant Activity Statistics', # 参与者活动统计
    'fall_ratio_red': 'Fall Ratio',          # 跌倒比例
    'data_quality_analysis': 'Data Quality - Missing Values', # 数据质量分析 - 缺失值
    'data_column': 'Data Column',            # 数据列
    'missing_ratio': 'Missing Ratio (%)',    # 缺失值比例 (%)
    'correlation_analysis': 'Sensor Correlation Analysis', # 传感器数据相关性分析
    'correlation_coefficient': 'Correlation Coefficient', # 相关系数
    'fall_type_distribution_pie': 'Fall Type Distribution', # 跌倒类型分布
    'fall_ratio_by_type': 'Fall Ratio by Type', # 各类型跌倒比例
    'participants_by_type': 'Participants by Type', # 各类型参与者数量
    'total_frames_by_type': 'Total Frames by Type', # 各类型总帧数
    'sensor_distribution_detailed': 'Distribution', # 分布
    'density': 'Density',                          # 密度
    'participant_detailed_analysis': 'Detailed Participant Analysis', # 参与者详细分析
    'participant_count': 'Participant Count',      # 参与者数
    'task_count': 'Task Count',                    # 任务数
    'trial_count': 'Trial Count',                  # 试验数
    'total_frames_distribution': 'Total Frames Distribution', # 参与者总帧数分布
    'fall_ratio_distribution': 'Fall Ratio Distribution',     # 参与者跌倒比例分布
    'task_count_distribution': 'Task Count Distribution',     # 参与者任务数分布
    'trial_count_distribution': 'Trial Count Distribution'    # 参与者试验数分布
}

class KFallVisualizer:
    def __init__(self, processed_data_path='processed_data/kfall_processed_data.csv', 
                 info_data_path='processed_data/kfall_data_info.csv'):
        """
        初始化数据可视化器
        
        Args:
            processed_data_path: 处理后的数据文件路径
            info_data_path: 数据信息文件路径
        """
        # 使用当前脚本所在目录作为基准
        script_dir = Path(__file__).parent
        self.processed_data_path = script_dir / processed_data_path
        self.info_data_path = script_dir / info_data_path
        
        # 加载数据
        self.load_data()
        
    def load_data(self):
        """加载处理后的数据"""
        if not self.processed_data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.processed_data_path}")
        
        print("正在加载数据...")
        self.df = pd.read_csv(self.processed_data_path)
        self.info_df = pd.read_csv(self.info_data_path)
        
        print(f"数据加载完成！")
        print(f"总数据量: {len(self.df):,} 行")
        print(f"参与者数量: {self.df['participant_id'].nunique()}")
        print(f"跌倒帧数: {self.df['label'].sum():,}")
        print(f"非跌倒帧数: {len(self.df) - self.df['label'].sum():,}")
    
    def create_comprehensive_visualization(self):
        """
        创建综合可视化图表
        """
        # 创建一个大图，包含多个子图
        fig = plt.figure(figsize=(20, 24))
        
        # 1. 数据分布概览
        self.plot_data_overview(fig, 3, 3, 1)
        
        # 2. 参与者数据分布
        self.plot_participant_distribution(fig, 3, 3, 2)
        
        # 3. 跌倒类型分布
        self.plot_fall_type_distribution(fig, 3, 3, 3)
        
        # 4. 传感器数据分布
        self.plot_sensor_distribution(fig, 3, 3, 4)
        
        # 5. 时间序列示例
        self.plot_time_series_example(fig, 3, 3, 5)
        
        # 6. 跌倒检测统计
        self.plot_fall_detection_stats(fig, 3, 3, 6)
        
        # 7. 参与者活动统计
        self.plot_participant_activity_stats(fig, 3, 3, 7)
        
        # 8. 数据质量分析
        self.plot_data_quality_analysis(fig, 3, 3, 8)
        
        # 9. 相关性分析
        self.plot_correlation_analysis(fig, 3, 3, 9)
        
        plt.tight_layout()
        output_dir = Path('visualization_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_dir / 'kfall_comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ 综合分析图生成完成")
        
        # 生成详细分析图表
        self.create_detailed_charts('visualization_results')
        
        print("\n所有图表生成完成！")
        print("生成的文件包括：")
        print("1. visualization_results/kfall_comprehensive_analysis.png - 综合分析图")
        print("2. visualization_results/fall_type_analysis.png - 跌倒类型分析")
        print("3. visualization_results/sensor_distribution_detailed.png - 传感器分布详细分析")
        print("4. visualization_results/participant_detailed_analysis.png - 参与者详细分析")
    
    def plot_data_overview(self, fig, rows, cols, pos):
        """数据概览图"""
        ax = plt.subplot(rows, cols, pos)
        
        # 标签分布饼图
        labels = [LABELS['non_fall'], LABELS['fall']]
        sizes = [len(self.df) - self.df['label'].sum(), self.df['label'].sum()]
        colors = ['#ff9999', '#66b3ff']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                         startangle=90, textprops={'fontsize': 10})
        ax.set_title(LABELS['data_overview'], fontsize=14, fontweight='bold')
        
        # 添加统计信息
        info_text = f'{LABELS["total_data"]}: {len(self.df):,}\n{LABELS["participants"]}: {self.df["participant_id"].nunique()}\n{LABELS["trials"]}: {len(self.info_df)}'
        ax.text(0, -1.5, info_text, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=10)
    
    def plot_participant_distribution(self, fig, rows, cols, pos):
        """参与者数据分布"""
        ax = plt.subplot(rows, cols, pos)
        
        # 统计每个参与者的数据量
        participant_stats = self.df.groupby('participant_id').agg({
            'label': ['count', 'sum', 'mean']
        }).round(4)
        participant_stats.columns = [LABELS['total_frames'], LABELS['fall_frames'], LABELS['fall_ratio']]
        
        # 绘制柱状图
        x = range(len(participant_stats))
        ax.bar(x, participant_stats[LABELS['total_frames']], alpha=0.7, color='skyblue', label=LABELS['total_frames'])
        ax.bar(x, participant_stats[LABELS['fall_frames']], alpha=0.9, color='red', label=LABELS['fall_frames'])
        
        ax.set_xlabel(LABELS['participant_id'], fontsize=12)
        ax.set_ylabel(LABELS['frame_count'], fontsize=12)
        ax.set_title(LABELS['participant_distribution'], fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 设置x轴标签 - 减少标签密度，避免重合
        step = max(1, len(participant_stats) // 6)  # 最多显示6个标签
        ax.set_xticks(x[::step])
        ax.set_xticklabels(participant_stats.index[::step], rotation=45, ha='right', fontsize=10)
    
    def plot_fall_type_distribution(self, fig, rows, cols, pos):
        """跌倒类型分布"""
        ax = plt.subplot(rows, cols, pos)
        
        # 统计跌倒类型
        fall_type_stats = self.df.groupby('fall_description').agg({
            'label': ['count', 'sum', 'mean']
        }).round(4)
        fall_type_stats.columns = [LABELS['total_frames'], LABELS['fall_frames'], LABELS['fall_ratio']]
        
        # 绘制水平柱状图
        y_pos = range(len(fall_type_stats))
        ax.barh(y_pos, fall_type_stats[LABELS['fall_frames']], color='lightcoral')
        
        ax.set_yticks(y_pos)
        # 对长标签进行换行处理
        wrapped_labels = wrap_labels(fall_type_stats.index, width=20)
        ax.set_yticklabels(wrapped_labels, fontsize=9)
        ax.set_xlabel(LABELS['fall_frames_count'], fontsize=12)
        ax.set_title(LABELS['fall_type_distribution'], fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def plot_sensor_distribution(self, fig, rows, cols, pos):
        """传感器数据分布"""
        ax = plt.subplot(rows, cols, pos)
        
        # 选择传感器列
        sensor_cols = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'EulerX', 'EulerY', 'EulerZ']
        
        # 计算每个传感器的统计信息
        sensor_stats = []
        for col in sensor_cols:
            sensor_stats.append({
                '传感器': col,
                '均值': self.df[col].mean(),
                '标准差': self.df[col].std(),
                '最小值': self.df[col].min(),
                '最大值': self.df[col].max()
            })
        
        sensor_df = pd.DataFrame(sensor_stats)
        
        # 绘制箱线图
        sensor_data = [self.df[col] for col in sensor_cols]
        bp = ax.boxplot(sensor_data, labels=sensor_cols, patch_artist=True)
        
        # 设置颜色
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 
                 'lightpink', 'lightgray', 'lightcyan', 'lightsteelblue', 'lightseagreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_xlabel(LABELS['sensor_type'], fontsize=12)
        ax.set_ylabel(LABELS['value'], fontsize=12)
        ax.set_title(LABELS['sensor_distribution'], fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.grid(True, alpha=0.3)
    
    def plot_time_series_example(self, fig, rows, cols, pos):
        """时间序列示例"""
        ax = plt.subplot(rows, cols, pos)
        
        # 选择一个包含跌倒的试验作为示例
        fall_example = self.df[self.df['label'] == 1].iloc[0]
        participant_id = fall_example['participant_id']
        task_id = fall_example['task_id']
        trial_id = fall_example['trial_id']
        
        # 获取该试验的所有数据
        example_data = self.df[
            (self.df['participant_id'] == participant_id) & 
            (self.df['task_id'] == task_id) & 
            (self.df['trial_id'] == trial_id)
        ].copy()
        
        # 绘制时间序列
        x = range(len(example_data))
        ax.plot(x, example_data['AccX'], label='AccX', alpha=0.7, linewidth=1)
        ax.plot(x, example_data['AccY'], label='AccY', alpha=0.7, linewidth=1)
        ax.plot(x, example_data['AccZ'], label='AccZ', alpha=0.7, linewidth=1)
        
        # 标记跌倒区域
        fall_mask = example_data['label'] == 1
        if fall_mask.any():
            fall_indices = np.where(fall_mask)[0]
            ax.axvspan(fall_indices[0], fall_indices[-1], alpha=0.3, color='red', label=LABELS['fall_region'])
        
        ax.set_xlabel(LABELS['frame_number'], fontsize=12)
        ax.set_ylabel(LABELS['acceleration'], fontsize=12)
        title_text = f'{LABELS["time_series_example"]}\n({participant_id} T{int(task_id):02d}{trial_id})'
        ax.set_title(title_text, fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def plot_fall_detection_stats(self, fig, rows, cols, pos):
        """跌倒检测统计"""
        ax = plt.subplot(rows, cols, pos)
        
        # 统计每个试验的跌倒检测情况
        trial_stats = self.info_df.copy()
        
        # 绘制散点图：总帧数 vs 跌倒帧数
        scatter = ax.scatter(trial_stats['total_frames'], trial_stats['fall_frames'], 
                  alpha=0.6, s=30, c=trial_stats['fall_ratio'], cmap='viridis')
        
        ax.set_xlabel(LABELS['total_frames_x'], fontsize=12)
        ax.set_ylabel(LABELS['fall_frames_y'], fontsize=12)
        ax.set_title(LABELS['fall_detection_stats'], fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(LABELS['fall_ratio'], fontsize=10)
    
    def plot_participant_activity_stats(self, fig, rows, cols, pos):
        """参与者活动统计"""
        ax = plt.subplot(rows, cols, pos)
        
        # 统计每个参与者的活动情况
        participant_activity = self.df.groupby('participant_id').agg({
            'label': ['count', 'sum', 'mean'],
            'task_id': 'nunique'
        }).round(4)
        participant_activity.columns = [LABELS['total_frames'], LABELS['fall_frames'], LABELS['fall_ratio'], '任务数']
        
        # 绘制双轴图
        ax2 = ax.twinx()
        
        bars = ax.bar(range(len(participant_activity)), participant_activity[LABELS['total_frames']], 
                     alpha=0.7, color='skyblue', label=LABELS['total_frames'])
        line = ax2.plot(range(len(participant_activity)), participant_activity[LABELS['fall_ratio']], 
                       'ro-', linewidth=2, markersize=4, label=LABELS['fall_ratio_red'])
        
        ax.set_xlabel(LABELS['participant_id'], fontsize=12)
        ax.set_ylabel(LABELS['total_frames'], color='blue', fontsize=12)
        ax2.set_ylabel(LABELS['fall_ratio'], color='red', fontsize=12)
        ax.set_title(LABELS['participant_activity_stats'], fontsize=14, fontweight='bold')
        
        # 设置x轴标签 - 减少密度
        step = max(1, len(participant_activity) // 5)
        ax.set_xticks(range(0, len(participant_activity), step))
        ax.set_xticklabels(participant_activity.index[::step], rotation=45, ha='right', fontsize=9)
        
        ax.grid(True, alpha=0.3)
    
    def plot_data_quality_analysis(self, fig, rows, cols, pos):
        """数据质量分析"""
        ax = plt.subplot(rows, cols, pos)
        
        # 检查缺失值
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        # 绘制缺失值统计
        bars = ax.bar(range(len(missing_data)), missing_percent, color='lightcoral')
        ax.set_xlabel(LABELS['data_column'], fontsize=12)
        ax.set_ylabel(LABELS['missing_ratio'], fontsize=12)
        ax.set_title(LABELS['data_quality_analysis'], fontsize=14, fontweight='bold')
        
        # 设置x轴标签
        ax.set_xticks(range(len(missing_data)))
        ax.set_xticklabels(missing_data.index, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(missing_percent):
            if v > 0:
                ax.text(i, v + 0.1, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    
    def plot_correlation_analysis(self, fig, rows, cols, pos):
        """相关性分析"""
        ax = plt.subplot(rows, cols, pos)
        
        # 选择数值列进行相关性分析
        numeric_cols = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'EulerX', 'EulerY', 'EulerZ', 'label']
        correlation_matrix = self.df[numeric_cols].corr()
        
        # 绘制热力图
        im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # 添加数值标签 - 减少字体大小
        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix)):
                text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=7)
        
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45, fontsize=8)
        ax.set_yticklabels(correlation_matrix.columns, fontsize=8)
        ax.set_title(LABELS['correlation_analysis'], fontsize=14, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(LABELS['correlation_coefficient'], fontsize=10)
    
    def create_detailed_charts(self, output_path):
        """创建详细的分析图表"""
        print("正在生成详细分析图表...")
        
        # 确保输出目录存在
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # 创建多个图表文件
        charts = [
            self.create_fall_type_pie_chart,
            self.create_sensor_distribution_chart,
            self.create_participant_analysis_chart
        ]
        
        for i, chart_func in enumerate(charts, 1):
            try:
                chart_func(output_path)
                print(f"✓ 图表 {i} 生成完成")
            except Exception as e:
                print(f"✗ 图表 {i} 生成失败: {e}")
    
    def create_fall_type_pie_chart(self, output_path):
        """创建跌倒类型饼图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 跌倒类型分布饼图
        fall_type_stats = self.df.groupby('fall_description').agg({
            'label': ['count', 'sum', 'mean']
        }).round(4)
        fall_type_stats.columns = [LABELS['total_frames'], LABELS['fall_frames'], LABELS['fall_ratio']]
        
        # 饼图1：跌倒比例
        labels = wrap_labels(fall_type_stats.index, width=15)
        sizes = fall_type_stats[LABELS['fall_frames']]
        colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                         startangle=90, textprops={'fontsize': 9})
        ax1.set_title(LABELS['fall_type_distribution_pie'], fontsize=16, fontweight='bold')
        
        # 饼图2：参与者数量
        participant_counts = self.df.groupby('fall_description')['participant_id'].nunique()
        labels2 = wrap_labels(participant_counts.index, width=15)
        sizes2 = participant_counts.values
        colors2 = plt.cm.Pastel1(np.linspace(0, 1, len(sizes2)))
        
        wedges2, texts2, autotexts2 = ax2.pie(sizes2, labels=labels2, colors=colors2, autopct='%1.1f%%', 
                                            startangle=90, textprops={'fontsize': 9})
        ax2.set_title(LABELS['participants_by_type'], fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(str(Path(output_path) / 'fall_type_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_sensor_distribution_chart(self, output_path):
        """创建传感器分布图"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        sensor_cols = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'EulerX', 'EulerY', 'EulerZ']
        
        for i, col in enumerate(sensor_cols):
            ax = axes[i]
            
            # 分别绘制跌倒和非跌倒的分布
            fall_data = self.df[self.df['label'] == 1][col]
            non_fall_data = self.df[self.df['label'] == 0][col]
            
            # 绘制直方图
            ax.hist(non_fall_data, bins=50, alpha=0.7, label=LABELS['non_fall'], color='blue', density=True)
            ax.hist(fall_data, bins=50, alpha=0.7, label=LABELS['fall'], color='red', density=True)
            
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel(LABELS['density'], fontsize=12)
            ax.set_title(f'{col} {LABELS["sensor_distribution_detailed"]}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(str(Path(output_path) / 'sensor_distribution_detailed.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_participant_analysis_chart(self, output_path):
        """创建参与者详细分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 统计参与者信息
        participant_stats = self.df.groupby('participant_id').agg({
            'label': ['count', 'sum', 'mean'],
            'task_id': 'nunique',
            'trial_id': 'nunique'
        }).round(4)
        participant_stats.columns = [LABELS['total_frames'], LABELS['fall_frames'], LABELS['fall_ratio'], 
                                   LABELS['task_count'], LABELS['trial_count']]
        
        # 子图1：总帧数分布
        ax1 = axes[0, 0]
        ax1.hist(participant_stats[LABELS['total_frames']], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel(LABELS['total_frames'], fontsize=12)
        ax1.set_ylabel(LABELS['participant_count'], fontsize=12)
        ax1.set_title(LABELS['total_frames_distribution'], fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 子图2：跌倒比例分布
        ax2 = axes[0, 1]
        ax2.hist(participant_stats[LABELS['fall_ratio']], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel(LABELS['fall_ratio'], fontsize=12)
        ax2.set_ylabel(LABELS['participant_count'], fontsize=12)
        ax2.set_title(LABELS['fall_ratio_distribution'], fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 子图3：任务数分布
        ax3 = axes[1, 0]
        task_counts = participant_stats[LABELS['task_count']].value_counts().sort_index()
        ax3.bar(task_counts.index, task_counts.values, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_xlabel(LABELS['task_count'], fontsize=12)
        ax3.set_ylabel(LABELS['participant_count'], fontsize=12)
        ax3.set_title(LABELS['task_count_distribution'], fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 子图4：试验数分布
        ax4 = axes[1, 1]
        trial_counts = participant_stats[LABELS['trial_count']].value_counts().sort_index()
        ax4.bar(trial_counts.index, trial_counts.values, alpha=0.7, color='lightyellow', edgecolor='black')
        ax4.set_xlabel(LABELS['trial_count'], fontsize=12)
        ax4.set_ylabel(LABELS['participant_count'], fontsize=12)
        ax4.set_title(LABELS['trial_count_distribution'], fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(str(Path(output_path) / 'participant_detailed_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """主函数"""
    try:
        # 创建可视化对象
        visualizer = KFallVisualizer()
        
        # 创建输出目录
        output_dir = Path('visualization_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成可视化图表
        visualizer.create_comprehensive_visualization()
        
        print("数据可视化完成！")
        print("生成的文件包括：")
        print("1. visualization_results/kfall_comprehensive_analysis.png - 综合分析图")
        print("2. visualization_results/fall_type_analysis.png - 跌倒类型分析")
        print("3. visualization_results/sensor_distribution_detailed.png - 传感器分布详细分析")
        print("4. visualization_results/participant_detailed_analysis.png - 参与者详细分析")
        
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        print("请确保数据预处理已完成，并且数据文件存在。")

if __name__ == "__main__":
    main() 
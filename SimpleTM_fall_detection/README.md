# SimpleTM 跌倒检测系统

本项目基于SimpleTM（简单时间序列模型）实现老年人跌倒检测功能。SimpleTM是一个高效的时序数据处理框架，特别适合检测传感器数据中的异常模式，如跌倒事件。

## 项目结构

- `models/`: 模型定义和实现
  - `simpletm_fall_detector.py`: SimpleTM跌倒检测模型实现
- `data_processors/`: 数据预处理模块
  - `data_processor.py`: 数据加载、预处理和分割实现
- `utils/`: 工具函数和辅助模块
  - `update_readme.py`: 自动更新README中的训练结果
- `experiments/`: 实验脚本和配置
- `train.py`: 模型训练脚本
- `evaluate.py`: 模型评估脚本
- `predict.py`: 模型预测脚本
- `final_script.py`: 一键执行全流程的脚本
- `train_with_time_series_splits.py`: 使用预分割时间序列数据集训练模型的专用脚本
- `requirements.txt`: 项目依赖

## 详细文件结构与内容说明

### 核心模型文件 (models/)

#### simpletm_fall_detector.py
这个文件包含SimpleTM跌倒检测模型的核心实现，主要内容包括：

- **Config类**：模型配置类，存储所有模型超参数
- **WaveletEmbedding类**：实现小波变换嵌入，提取多尺度特征
  - 支持自动调整卷积参数，确保维度匹配
  - 使用PyWavelets库进行离散小波变换
- **GeomAttentionLayer类**：几何注意力层实现
  - 结合点积注意力和距离感知注意力
  - 使用alpha参数平衡两种注意力机制的权重
  - 支持处理3维和4维输入，增强了兼容性
- **SimpleTM_FallDetector类**：完整的跌倒检测模型
  - 包含小波嵌入层、几何注意力层和分类头
  - 支持返回注意力权重用于可视化和解释

### 数据处理模块 (data_processors/)

#### data_processor.py
包含数据加载、预处理和特征提取的核心函数：

- **SensorDataProcessor类**：传感器数据处理类
  - `load_data`：从CSV文件加载传感器数据
  - `preprocess`：清洗、填充缺失值和标准化
  - `create_windows`：滑动窗口特征提取
  - `process_for_simpleTM`：调整数据格式以适配SimpleTM模型
- **prepare_data_for_simpleTM函数**：完整的数据准备流程
  - 加载、预处理、分割和创建数据加载器
- **load_from_time_series_splits函数**：从预分割数据集加载数据
  - 直接加载train.csv、val.csv和test.csv文件
  - 处理预分割数据并创建数据加载器
- **create_dataloaders函数**：创建PyTorch数据加载器

### 训练与评估脚本

#### train.py
模型训练的主要脚本：

- **parse_args函数**：解析命令行参数
  - 支持`--use_splits`参数指定使用预分割数据集
  - 支持`--splits_dir`参数指定预分割数据集目录
- **train_epoch函数**：单个训练周期的实现
- **evaluate函数**：验证集评估函数
- **save_confusion_matrix函数**：保存混淆矩阵可视化
- **plot_training_history函数**：绘制训练历史曲线
- **main函数**：训练主流程
  - 根据参数选择数据加载方式（原始数据或预分割数据集）
  - 初始化模型、优化器和学习率调度器
  - 实现早停机制和模型保存

#### evaluate.py
模型评估脚本：

- **parse_args函数**：与train.py类似，支持预分割数据集
- **visualize_attention函数**：可视化注意力权重
- **save_confusion_matrix函数**：保存混淆矩阵
- **evaluate_model函数**：在测试集上评估模型
- **main函数**：评估主流程
  - 支持从预分割数据集加载测试数据
  - 加载预训练模型并进行评估
  - 生成混淆矩阵和分类报告

#### predict.py
用于新数据预测的脚本：

- **parse_args函数**：解析命令行参数
- **TestDataset类**：用于预测的PyTorch数据集
- **preprocess_data函数**：预处理输入数据
- **visualize_predictions函数**：可视化预测结果
- **main函数**：预测主流程
  - 加载测试数据和预训练模型
  - 进行预测并输出结果
  - 可选生成可视化图表

### 一键执行脚本

#### final_script.py
集成训练、评估和预测的完整流程：

- **parse_args函数**：解析命令行参数
  - 支持`--use_splits`和`--splits_dir`参数
  - 支持`--skip_train`、`--skip_eval`和`--skip_predict`选项
- **run_command函数**：执行shell命令并捕获输出
- **main函数**：主流程
  - 训练阶段：根据参数选择数据源，调用train.py
  - 评估阶段：调用evaluate.py评估模型
  - 预测阶段：调用predict.py对新数据进行预测
  - 可选更新README.md中的训练结果

#### train_with_time_series_splits.py
专门用于预分割数据集训练的脚本：

- **parse_args函数**：解析命令行参数
  - 支持窗口大小、步长、批次大小等参数
- **main函数**：主流程
  - 构建train.py命令行参数并执行
  - 构建evaluate.py命令行参数并执行
  - 自动创建结果目录和模型保存目录

### 其他文件

#### requirements.txt
项目依赖包列表：

```
torch>=1.8.0
numpy>=1.19.5
pandas>=1.1.5
scikit-learn>=0.24.2
matplotlib>=3.3.4
seaborn>=0.11.1
pywavelets>=1.1.1
joblib>=1.0.1
```

#### checkpoints/
模型权重保存目录：
- `best_model.pth`：训练得到的最佳模型权重
- `scaler.joblib`：数据标准化器

#### results/
结果保存目录：
- `training_history.png`：训练历史曲线
- `confusion_matrix.png`：混淆矩阵
- `classification_report.txt`：分类报告详情
- `evaluation/`：评估结果目录
  - `attention_maps/`：注意力可视化图

## 系统架构

### 数据处理流程

1. **数据加载**: 
   - 从CSV文件加载传感器数据
   - 支持直接从预分割的数据集加载（train.csv, val.csv, test.csv）
2. **预处理**: 清洗、填充缺失值、标准化
3. **窗口切分**: 使用滑动窗口方法提取时间序列特征
4. **数据转换**: 将数据转换为SimpleTM模型需要的格式 (batch_size, n_features, seq_len)

### 模型架构

SimpleTM 跌倒检测模型结合了以下关键技术:

1. **小波变换嵌入**: 提取多尺度时间序列特征
2. **几何注意力机制**: 结合点积和楔形积信息，增强对时序模式的感知
3. **轻量级编码器**: 单层设计减少计算复杂度
4. **分类头**: 使用全连接层将特征映射到跌倒/非跌倒概率

### 数据流图

```
输入传感器数据 (CSV)
     │
     ▼
   预处理
 (填充、标准化)
     │
     ▼
  滑动窗口分割
     │
     ▼
 数据格式转换
 [样本数,特征数,序列长度]
     │
     ▼
  小波变换嵌入
     │
     ▼
  几何注意力层
     │
     ▼
   分类器头部
     │
     ▼
 跌倒/非跌倒预测
```

## 安装与使用

### 环境要求

- Python 3.7+
- PyTorch 1.8.0+
- CUDA (可选，用于GPU加速)
- 其他依赖: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, PyWavelets

### 安装依赖

```bash
# 安装依赖
pip install -r requirements.txt
```

### 数据准备

确保数据格式正确：
- CSV格式
- 包含传感器数据列（加速度、角速度等）
- 可选的标签列（0表示正常活动，1表示跌倒）

支持两种数据加载方式：
1. **原始数据**：单个CSV文件，系统会自动分割为训练/验证/测试集
2. **预分割数据集**：包含train.csv、val.csv、test.csv的目录，直接加载预分割好的数据

### 模型训练

```bash
# 基本训练（使用原始数据）
python train.py

# 使用预分割数据集训练
python train.py --use_splits --splits_dir "../KFall Dataset/time_series_splits"

# 使用专用脚本训练（预分割数据集）
python train_with_time_series_splits.py

# 自定义训练参数
python train.py --data_path <数据路径> --window_size 128 --stride 32 --batch_size 64 --epochs 50 --d_model 256 --dropout 0.2 --alpha 0.5
```

主要参数说明：
- `--data_path`: 传感器数据文件路径
- `--use_splits`: 是否使用预分割的数据集
- `--splits_dir`: 预分割数据集目录
- `--window_size`: 滑动窗口大小（时间步长）
- `--stride`: 滑动窗口步长
- `--batch_size`: 训练批次大小
- `--epochs`: 训练轮数
- `--d_model`: 模型维度
- `--dropout`: Dropout比率
- `--alpha`: 几何注意力中的平衡系数（0-1之间）

### 模型评估

```bash
# 评估已训练模型（原始数据）
python evaluate.py --model_path ./checkpoints/best_model.pth

# 评估已训练模型（预分割数据集）
python evaluate.py --use_splits --splits_dir "../KFall Dataset/time_series_splits" --model_path ./checkpoints/best_model.pth

# 可视化注意力权重
python evaluate.py --model_path ./checkpoints/best_model.pth --save_attention
```

主要参数说明：
- `--model_path`: 模型权重路径
- `--data_path`: 用于评估的数据文件路径
- `--use_splits`: 是否使用预分割的数据集
- `--splits_dir`: 预分割数据集目录
- `--save_attention`: 是否保存注意力可视化图

### 模型预测

```bash
# 对新数据进行预测
python predict.py --input_file <输入文件路径> --model_path ./checkpoints/best_model.pth --visualize
```

主要参数说明：
- `--input_file`: 输入数据文件路径
- `--model_path`: 模型权重路径
- `--output_file`: 预测结果保存路径
- `--visualize`: 是否可视化预测结果

### 一键执行全流程

```bash
# 执行全部流程（训练、评估、预测）- 原始数据
python final_script.py

# 使用预分割数据集执行全流程
python final_script.py --use_splits --splits_dir "../KFall Dataset/time_series_splits"

# 跳过训练，仅执行评估和预测
python final_script.py --skip_train --test_file <测试文件路径>

# 更新README.md中的训练结果
python final_script.py --update_readme
```

主要参数说明：
- `--use_splits`: 是否使用预分割的数据集
- `--splits_dir`: 预分割数据集目录
- `--skip_train`: 跳过训练阶段
- `--skip_eval`: 跳过评估阶段
- `--skip_predict`: 跳过预测阶段
- `--test_file`: 用于预测的测试文件路径
- `--update_readme`: 更新README.md中的训练结果

### 使用预分割数据集专用训练脚本

我们提供了专门用于使用预分割数据集训练和评估的脚本：

```bash
# 基本用法
python train_with_time_series_splits.py

# 自定义参数
python train_with_time_series_splits.py --window_size 128 --stride 32 --batch_size 64 --epochs 50 --d_model 256
```

主要参数说明：
- `--splits_dir`: 预分割数据集目录
- `--window_size`: 滑动窗口大小
- `--stride`: 滑动窗口步长
- `--batch_size`: 训练批次大小
- `--epochs`: 训练轮数
- 其他模型参数与train.py相同

## 训练结果

| 模型配置 | 准确率 | 精确率 | 召回率 | F1分数 |
|---------|-------|-------|--------|--------|
| SimpleTM (原始) | - | - | - | - |
| SimpleTM (w=128, d=256, α=0.5) | - | - | - | - |
| SimpleTM (time_series_splits) | 92.92% | 92.85% | 92.92% | 92.87% |

*注: time_series_splits模型基于预分割数据集训练，使用参数：window_size=128, d_model=256, dropout=0.2, alpha=0.5*

### 详细性能指标 (time_series_splits模型)

**类别性能：**
- Normal类 (非跌倒): 精确率=94.41%, 召回率=95.90%, F1=95.15%
- Fall类 (跌倒): 精确率=88.74%, 召回率=85.07%, F1=86.87%

**评估指标解释：**
- 准确率(Accuracy): 所有预测中正确的比例
- 精确率(Precision): 预测为跌倒的样本中真正为跌倒的比例
- 召回率(Recall): 所有真实跌倒样本中被正确识别的比例
- F1分数: 精确率和召回率的调和平均值

### 混淆矩阵

测试集上的混淆矩阵已保存在`./results/time_series_split_model/confusion_matrix.png`，直观展示了模型的预测性能。

![混淆矩阵](./results/time_series_split_model/confusion_matrix.png)

### 训练历史曲线

训练过程中的损失和指标曲线已保存在`./results/time_series_split_model/training_history.png`，展示了模型在训练过程中的性能变化。

![训练历史](./results/time_series_split_model/training_history.png)

### 注意力可视化

模型的注意力权重可视化结果保存在`./results/time_series_split_model/evaluation/attention_maps/`，展示了模型关注的时间点和特征。

## 常见问题与解决方案

### 数据处理问题

1. **非数值列处理错误**
   - 症状: `ValueError: could not convert string to float`
   - 解决: 数据处理器现已增强，能够自动识别并排除非数值列

2. **维度不匹配**
   - 症状: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`
   - 解决: SimpleDataEmbedding类现支持动态调整特征维度
   
3. **加载预分割数据集错误**
   - 症状: `FileNotFoundError: 无法找到预分割数据集文件`
   - 解决: 确保splits_dir路径正确，并且包含train.csv、val.csv和test.csv文件

### 模型训练问题

1. **ReduceLROnPlateau参数错误**
   - 症状: `TypeError: __init__() got an unexpected keyword argument`
   - 解决: 增加了对不同PyTorch版本的兼容性支持

2. **小波变换卷积错误**
   - 症状: `RuntimeError: group的输入通道数必须能被group数整除`
   - 解决: 增强了WaveletEmbedding类自动调整卷积参数的能力
   
3. **GeomAttention维度错误**
   - 症状: `RuntimeError: Expected 4-dimensional input, got 3-dimensional`
   - 解决: 修改了GeomAttention类，现支持处理3维和4维输入

### 性能优化

1. **模型过拟合**
   - 解决: 尝试增加dropout率，或降低模型维度(d_model)

2. **检测准确率不高**
   - 解决: 尝试调整几何注意力系数alpha，或增加编码器层数
   
3. **预分割数据集训练效果不佳**
   - 解决: 检查预分割数据集的平衡性，可尝试使用--batch_size调整批次大小或使用不同的学习率

## 数据集

本项目使用传感器采集的加速度、角速度等数据进行跌倒检测。数据集包括多名老年人在日常活动和模拟跌倒场景下的传感器数据。

支持的数据集格式：
1. 原始传感器数据CSV
2. 预分割的时间序列数据集（train.csv, val.csv, test.csv）

## 最近更新

- 添加了对预分割时间序列数据集的支持
- 优化了GeomAttention层，支持处理不同维度的输入
- 增强了WaveletEmbedding类的稳定性
- 添加了专用于预分割数据集训练的脚本
- 统一了各模块的命令行接口

## 贡献与改进

欢迎提交问题和改进建议。若要贡献代码，请先创建issue讨论您的改动。

## 参考文献

- SimpleTM论文: [SimpleTM: A Simple but Effective Transformer-based Time Series Model](https://openreview.net/pdf?id=oANkBaVci5)
- SimpleTM代码库: [SimpleTM GitHub Repository](https://github.com/chenhui118/SimpleTM)

如果您使用了本代码，请同时引用SimpleTM原论文:

```bibtex
@inproceedings{
chen2025simpletm,
title={Simple{TM}: A Simple Baseline for Multivariate Time Series Forecasting},
author={Hui Chen and Viet Luong and Lopamudra Mukherjee and Vikas Singh},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=oANkBaVci5}
}
```

## 致谢

本项目基于SimpleTM框架开发，感谢原作者Hui Chen等人的开源贡献。我们对原始框架进行了修改，使其适用于跌倒检测任务，并添加了对预分割时间序列数据集的支持。
# KFall Dataset - 老人跌倒检测系统

本项目使用KFall数据集实现老人跌倒检测和预测系统，包含数据预处理、特征提取、模型训练和评估等多个模块。

## 文件结构

### 数据处理脚本

- **`data_preprocessing.py`**: 原始数据预处理脚本，将传感器数据与标签数据结合，生成处理后的数据集
- **`data_reclassification.py`**: 对数据进行重新分类，将活动分为坐姿和走路两大类
- **`data_split.py`**: 创建训练集、验证集和测试集的划分，使用滑动窗口生成样本
- **`data_visualization.py`**: 数据可视化脚本，生成各种图表分析数据集特征
- **`time_series_split.py`**: 创建保留时间序列连续性的数据集划分

### 时间序列平衡脚本

- **`balanced_time_series.py`**: 生成平衡的时间序列数据集，保持时间连续性，但不改变数据内部结构
- **`balanced_time_series_sampling.py`**: 通过对非跌倒帧进行采样，生成跌倒与非跌倒比例约为1:2的平衡数据集，同时保持时间连续性

### 模型训练脚本

- **`fall_prediction_lstm.py`**: 原始LSTM模型训练脚本，使用time_series_splits中的数据
- **`lstm_fall_prediction_balanced.py`**: 使用平衡后的数据集训练LSTM模型，跌倒与非跌倒比例约为1:2
- **`lstm_fall_prediction_balanced_regularized.py`**: 添加正则化、批标准化和注意力机制的LSTM模型
- **`gru_fall_prediction_balanced.py`**: 使用GRU代替LSTM的模型实现
- **`bidirectional_gru_fall_prediction.py`**: 使用双向GRU架构的跌倒预测模型
- **`cnn_lstm_fall_prediction.py`**: 结合1D-CNN和LSTM的混合模型
- **`cnn_gru_fall_prediction.py`**: 结合1D-CNN和GRU的混合模型
- **`tcn_fall_prediction.py`**: 使用时间卷积网络(TCN)的跌倒预测模型
- **`svm_fall_detection.py`**: 使用SVM进行跌倒检测的基础版本
- **`svm_fall_detection_improved.py`**: 改进版SVM跌倒检测模型
- **`tune_threshold_unbalanced.py`**: 不平衡数据集上CNN-LSTM模型的阈值寻优脚本
- **`fine_tune_threshold.py`**: 不平衡数据集上CNN-LSTM模型的精细阈值调优脚本

### 数据文件夹

- **`sensor_data/`**: 原始传感器数据，按参与者ID组织
- **`label_data/`**: 原始标签数据，包含跌倒发生的时间戳
- **`processed_data/`**: 经过预处理后的数据，包括：
  - `kfall_processed_data.csv`: 处理后的完整数据集
  - `kfall_data_info.csv`: 数据集信息摘要
- **`reclassified_data/`**: 重新分类后的数据，包括：
  - `kfall_reclassified_data.csv`: 包含活动类型分类的完整数据
  - `sitting_activities.csv`: 坐姿活动数据
  - `walking_activities.csv`: 走路活动数据
  - `reclassification_info.csv`: 重分类信息
- **`split_data/`**: 按活动类型划分的训练、验证和测试集
- **`time_series_splits/`**: 保留时间序列连续性的数据集划分
- **`balanced_time_series_splits/`**: 平衡的时间序列数据集，保持原始比例
- **`balanced_time_series_splits_1to2/`**: 跌倒与非跌倒比例约为1:2的平衡数据集

### 模型和结果文件

- **`best_model.pth`**: 原始LSTM模型训练的最佳权重
- **`best_model_balanced.pth`**: 使用平衡数据集训练的LSTM模型最佳权重
- **`best_model_lstm_regularized.pth`**: 添加正则化的LSTM模型权重
- **`best_model_gru.pth`**: GRU模型的最佳权重
- **`best_model_bidirectional_gru.pth`**: 双向GRU模型的最佳权重
- **`best_model_cnn_lstm.pth`**: CNN-LSTM混合模型的最佳权重
- **`best_model_cnn_gru.pth`**: CNN-GRU混合模型的最佳权重
- **`best_model_tcn.pth`**: TCN模型的最佳权重
- **`best_model_gru_unbalanced.pth`**: 未采样GRU模型的最佳权重，用于验证采样策略对预测时间的影响
- **`best_model_cnn_lstm_unbalanced.pth`**: 未采样CNN-LSTM模型的最佳权重，用于验证采样策略对预测时间的影响

- **`confusion_matrix.png`**: 原始模型的混淆矩阵
- **`confusion_matrix_balanced.png`**: 使用第一版平衡数据集的混淆矩阵
- **`confusion_matrix_balanced_1to2.png`**: 使用1:2平衡数据集的混淆矩阵
- **`confusion_matrix_lstm_regularized.png`**: 正则化LSTM模型的混淆矩阵
- **`confusion_matrix_gru.png`**: GRU模型的混淆矩阵
- **`confusion_matrix_bidirectional_gru.png`**: 双向GRU模型的混淆矩阵
- **`confusion_matrix_cnn_lstm.png`**: CNN-LSTM混合模型的混淆矩阵
- **`confusion_matrix_cnn_gru.png`**: CNN-GRU混合模型的混淆矩阵
- **`confusion_matrix_tcn.png`**: TCN模型的混淆矩阵
- **`confusion_matrix_gru_unbalanced.png`**: 未采样GRU模型的混淆矩阵
- **`confusion_matrix_cnn_lstm_unbalanced.png`**: 未采样CNN-LSTM模型的混淆矩阵
- **`confusion_matrix_cnnlstm_optimal.png`**: 使用最优阈值的CNN-LSTM模型混淆矩阵

- **`training_history.png`**: 原始模型的训练历史
- **`training_history_balanced.png`**: 使用第一版平衡数据集的训练历史
- **`training_history_balanced_1to2.png`**: 使用1:2平衡数据集的训练历史
- **`training_history_lstm_regularized.png`**: 正则化LSTM模型的训练历史
- **`training_history_gru.png`**: GRU模型的训练历史
- **`training_history_bidirectional_gru.png`**: 双向GRU模型的训练历史
- **`training_history_cnn_lstm.png`**: CNN-LSTM混合模型的训练历史
- **`training_history_cnn_gru.png`**: CNN-GRU混合模型的训练历史
- **`training_history_tcn.png`**: TCN模型的训练历史
- **`training_history_gru_unbalanced.png`**: 未采样GRU模型的训练历史
- **`training_history_cnn_lstm_unbalanced.png`**: 未采样CNN-LSTM模型的训练历史
- **`pr_curve_cnn_lstm_unbalanced.png`**: 未采样CNN-LSTM模型的PR曲线
- **`threshold_tuning_cnn_lstm_unbalanced.png`**: 未采样CNN-LSTM模型的阈值调优图
- **`threshold_tuning_cnn_lstm_unbalanced_fine_grained.png`**: 未采样CNN-LSTM模型的精细阈值调优图

### 其他文件

- **`requirements.txt`**: 项目依赖包列表
- **`simhei.ttf`**: 中文字体文件，用于图表显示中文

## 数据处理流程

1. **数据预处理**: 使用`data_preprocessing.py`将原始传感器数据与标签数据结合
2. **数据重分类**: 使用`data_reclassification.py`对活动进行分类
3. **数据划分**: 使用`data_split.py`或`time_series_split.py`创建数据集划分
4. **数据平衡**: 使用`balanced_time_series.py`或`balanced_time_series_sampling.py`创建平衡数据集
5. **模型训练**: 使用各种模型训练脚本训练不同架构的模型

## 模型比较

### 基础模型
- **原始LSTM模型**: 使用未平衡的数据集，跌倒与非跌倒比例约为1:9
- **平衡LSTM模型**: 使用平衡的数据集，跌倒与非跌倒比例约为1:2
- **SVM模型**: 使用传统机器学习方法的基准模型

### 改进模型
- **正则化LSTM模型**: 在基础LSTM上添加批标准化、Dropout和L2正则化
- **GRU模型**: 使用门控循环单元(GRU)代替LSTM，减少参数量并加快训练
- **双向GRU模型**: 使用双向架构捕获时间序列的双向依赖关系
- **CNN-LSTM混合模型**: 结合1D卷积网络和LSTM，先提取局部特征再捕获长期依赖
- **CNN-GRU混合模型**: 结合1D卷积网络和GRU，性能与CNN-LSTM相当但训练更快
- **TCN模型**: 使用时间卷积网络，通过扩张卷积和残差连接有效捕获长距离时间依赖

### 未采样模型
- **未采样GRU模型**: 使用原始未平衡数据集（不进行1:2采样）训练的GRU模型，用于验证采样策略对模型预测时间的影响
- **未采样CNN-LSTM模型**: 使用原始未平衡数据集训练的CNN-LSTM混合模型，保持所有超参数与平衡版本一致，仅改变数据集采样策略
- **目的**: 验证采样策略是否导致模型在实际跌倒发生前约2秒误判为跌倒，通过与采样模型进行对比分析预测时间差异

## 模型性能比较

| 模型 | 准确率 | AUC | 跌倒精确率 | 跌倒召回率 | 训练轮次 |
|------|--------|-----|------------|------------|----------|
| 原始LSTM | 91% | 0.9450 | 85% | 79% | 28 |
| 平衡LSTM | 94% | 0.9750 | 92% | 91% | 25 |
| 正则化LSTM | 96% | 0.9850 | 95% | 94% | 24 |
| GRU | 96% | 0.9860 | 95% | 95% | 20 |
| 双向GRU | 96% | 0.9870 | 96% | 94% | 19 |
| CNN-LSTM | 97% | 0.9933 | 97% | 98% | 22 |
| CNN-GRU | 97% | 0.9929 | 98% | 97% | 16 |
| TCN | 97% | 0.9901 | 96% | 98% | 13 |
| 未采样GRU | 92% | 0.9400 | 90% | 88% | 17 |
| 未采样CNN-LSTM | 93% | 0.9911 | 67% | 96% | 27 |
| 未采样CNN-LSTM (优化阈值0.84) | 93% | 0.9911 | 91% | 87% | 27 |

## 使用方法

### 数据预处理

```bash
python data_preprocessing.py
python data_reclassification.py
```

### 创建平衡数据集

```bash
python balanced_time_series_sampling.py
```

### 训练模型

```bash
# 训练基础LSTM模型
python lstm_fall_prediction_balanced.py

# 训练正则化LSTM模型
python lstm_fall_prediction_balanced_regularized.py

# 训练GRU模型
python gru_fall_prediction_balanced.py

# 训练双向GRU模型
python bidirectional_gru_fall_prediction.py

# 训练CNN-LSTM混合模型
python cnn_lstm_fall_prediction.py

# 训练CNN-GRU混合模型
python cnn_gru_fall_prediction.py

# 训练TCN模型
python tcn_fall_prediction.py

# 训练未采样GRU模型（验证采样策略影响）
python gru_fall_prediction_unbalanced.py

# 训练未采样CNN-LSTM模型（验证采样策略影响）
python cnn_lstm_fall_prediction_unbalanced.py

# 对未采样CNN-LSTM模型进行阈值调优
python tune_threshold_unbalanced.py

# 对未采样CNN-LSTM模型进行精细阈值调优
python fine_tune_threshold.py
```

### 实时预测系统

项目包含一个基于Flask和Socket.IO的实时预测仪表盘，可以可视化传感器数据和模型预测结果：

- 支持多种预训练模型的选择和切换，包括采样和未采样模型
- 实时显示9个传感器数据的波形图
- 提供跌倒风险实时预测和置信度显示
- 包含暂停/继续/停止等交互控制功能
- 可调整模拟速度，便于观察和分析
- 支持对比不同模型在预测时间上的差异

启动实时预测仪表盘：

```bash
# 启动实时预测仪表盘
cd realtime_dashboard
python app.py
```

实时预测系统位于`realtime_dashboard`目录，详细使用说明请参见该目录下的README文件。

## 结果分析

通过比较各个模型的训练历史和混淆矩阵，可以观察不同模型架构对跌倒检测性能的影响：

1. **数据平衡的影响**：平衡后的数据集(1:2)显著改善了模型的学习过程，验证损失更低，AUC更高。

2. **正则化的影响**：添加批标准化、Dropout和L2正则化后，模型泛化能力增强，过拟合减少。

3. **模型架构比较**：
   - GRU比LSTM训练更快，性能相当或略好
   - 双向GRU在捕获时间序列特征方面表现更好
   - CNN-LSTM和CNN-GRU混合模型通过结合卷积和循环网络的优势，达到了最佳性能
   - TCN模型训练速度最快，且在保持高准确率的同时实现了最高的跌倒召回率

4. **最佳模型选择**：
   - 对于实时性要求高的场景，推荐使用TCN或CNN-GRU模型
   - 对于准确率要求高的场景，推荐使用CNN-LSTM模型
   - 对于资源受限的环境，推荐使用GRU或双向GRU模型

5. **验证采样策略对预测时间的影响**：
   - 通过对比采样(1:2)和未采样模型的预测结果，分析采样策略是否导致预测时间提前
   - 未采样模型使用原始数据集的类别分布（约1:9），保持所有其他训练参数和模型架构不变
   - 混淆矩阵和训练历史分析显示：
     - 未采样GRU模型准确率为92%，AUC为0.94，跌倒召回率为88%，比采样GRU模型（准确率96%，AUC为0.986，跌倒召回率95%）性能略低
     - 未采样CNN-LSTM模型准确率为94%，AUC为0.96，跌倒召回率为92%，同样低于采样CNN-LSTM模型（准确率97%，AUC为0.993，跌倒召回率98%）
     - 未采样模型在真实跌倒事件的预测时间上与采样模型有显著差异，预测时间更接近实际跌倒发生时刻
     - 采样模型对跌倒前兆的敏感度更高，导致其在实际跌倒发生前约2秒即做出预测
   - 结论：1:2采样策略确实是导致模型在实际跌倒发生前约2秒做出预测的主要原因之一
   - 实际应用建议：
     - 对于早期预警系统，采用采样模型可提前约2秒预测跌倒风险
     - 对于精确时间戳要求高的系统，可采用未采样模型获得更接近实际跌倒时刻的预测
     - 可以根据不同应用场景的需求选择合适的模型训练策略

混淆矩阵显示了各模型在测试集上的性能，通过比较可以看出改进模型对预测性能的提升，特别是在减少假阴性(漏报跌倒)方面的改进。采样与未采样模型的对比直观地展示了数据采样策略对预测时间的影响，为开发针对不同应用场景的跌倒预测系统提供了重要参考。

### 未采样模型的实验结果详细分析

未采样模型与采样模型的对比实验证明了我们的假设：采样策略是影响模型预测时间的关键因素。详细分析如下：

1. **预测时间差异**：
   - 采样模型在实际跌倒发生前约2秒即可做出预测，提供了更长的预警时间
   - 未采样模型预测时间更接近实际跌倒发生时刻，延迟约1-1.5秒才发出预警
   - 在实时系统中，这种时间差异显著影响了干预措施的启动时机

2. **精确度和召回率权衡**：
   - 未采样模型虽然在整体准确率上略低，但在预测时间准确性上更高
   - 采样模型在跌倒检出率(召回率)上表现更好，但会产生更多的预测时间提前
   - 这种权衡关系说明在实际应用中需要根据预警时间和准确性需求选择合适模型

3. **数据分布与模型行为**：
   - 采样策略(1:2)使模型更多地接触跌倒样本，导致模型对跌倒前兆特征更敏感
   - 未采样模型训练在原始数据分布(1:9)上，需要更强的跌倒特征才会触发预警
   - 这种差异解释了两种模型在预测时间点上的系统性偏差

4. **实际应用场景适配**：
   - 针对老人护理中心等需要提前干预的场景，采样模型提供的2秒预警时间更有价值
   - 针对研究或记录目的，未采样模型提供的接近实时的跌倒检测更为准确
   - 结合两种模型可实现"双重预警系统"：先由采样模型提供早期预警，再由未采样模型确认实际跌倒时刻

5. **阈值调优对未采样模型的影响**：
   - 未采样CNN-LSTM模型在默认阈值(0.5)下显示"高召回率(0.96)、低精确率(0.67)"的特性
   - 通过精细阈值调优，在阈值为0.84时达到最佳F1分数(0.886)
   - 优化后模型的精确率提升至0.91，召回率降至0.87，大幅减少误报同时保持较高的跌倒检出率
   - 对于需要平衡精确率和召回率的应用场景，阈值0.79时两者均达到0.883

实时预测系统位于`realtime_dashboard`目录，详细使用说明请参见该目录下的README文件。 
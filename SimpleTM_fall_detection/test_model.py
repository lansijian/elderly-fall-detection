import os  
import sys  
import torch  
import numpy as np  
import traceback  
from models.simpletm_fall_detector import SimpleTM_FallDetector, Config  
  
print("测试SimpleTM模型")  
  
try:  
    # 测试数据维度
    batch_size = 4
    n_features = 9  # 输入特征数
    seq_len = 128   # 序列长度
    
    # 创建模型配置，确保dec_in参数与输入特征数匹配
    config = Config(
        seq_len=seq_len, 
        d_model=256, 
        dropout=0.2, 
        num_classes=2,
        dec_in=n_features,  # 设置dec_in与输入特征数匹配
        output_attention=False  # 设置为False以简化输出
    )
    
    # 创建模型
    model = SimpleTM_FallDetector(config)
    print('模型创建成功')
    
    # 创建测试数据 [batch_size, n_features, seq_len]
    x = torch.randn(batch_size, n_features, seq_len)
    print(f'输入形状: {x.shape}')
    
    # 前向传播
    output, l1_reg = model(x)
    print(f'输出形状: {output.shape}')
    print(f'L1正则化值: {l1_reg:.4f}')
    
    # 获取预测结果
    predictions = torch.argmax(output, dim=1)
    print(f'预测结果: {predictions}')
    print('测试成功！')
    
except Exception as e:
    print(f'错误: {e}')
    traceback.print_exc() 

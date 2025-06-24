import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time
import gc
import warnings
warnings.filterwarnings('ignore')

def load_and_concat(paths):
    """优化的数据加载函数"""
    dfs = []
    for p in tqdm(paths, desc='加载数据'):
        try:
            # 只读取需要的列，减少内存使用
            df = pd.read_csv(p, usecols=['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'EulerX', 'EulerY', 'EulerZ', 'label'])
            dfs.append(df)
        except Exception as e:
            print(f"加载文件 {p} 时出错: {e}")
            continue
    return pd.concat(dfs, ignore_index=True)

def main():
    start_time = time.time()
    
    # 数据路径
    sitting_train = 'KFall Dataset/split_data/sitting/train.csv'
    sitting_val = 'KFall Dataset/split_data/sitting/val.csv'
    sitting_test = 'KFall Dataset/split_data/sitting/test.csv'
    walking_train = 'KFall Dataset/split_data/walking/train.csv'
    walking_val = 'KFall Dataset/split_data/walking/val.csv'
    walking_test = 'KFall Dataset/split_data/walking/test.csv'
    
    print("=== SVM跌倒检测模型训练 ===")
    
    # 合并训练、验证、测试集
    print("\n1. 加载数据...")
    df_train = load_and_concat([sitting_train, walking_train])
    df_val = load_and_concat([sitting_val, walking_val])
    df_test = load_and_concat([sitting_test, walking_test])
    
    print(f"训练集大小: {len(df_train):,} 样本")
    print(f"验证集大小: {len(df_val):,} 样本")
    print(f"测试集大小: {len(df_test):,} 样本")
    
    # 选择特征和标签
    feature_cols = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'EulerX', 'EulerY', 'EulerZ']
    label_col = 'label'
    
    X_train = df_train[feature_cols].values
    y_train = df_train[label_col].values
    X_val = df_val[feature_cols].values
    y_val = df_val[label_col].values
    X_test = df_test[feature_cols].values
    y_test = df_test[label_col].values
    
    # 释放内存
    del df_train, df_val, df_test
    gc.collect()
    
    # 标准化
    print("\n2. 数据标准化...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # 训练SVM - 使用更高效的参数
    print("\n3. 训练SVM模型...")
    train_start = time.time()
    
    # 使用线性核函数，训练更快
    clf = SVC(
        kernel='linear',  # 线性核比RBF更快
        C=1.0,
        class_weight='balanced',
        random_state=42,
        probability=True,
        max_iter=1000,  # 限制最大迭代次数
        cache_size=1000  # 增加缓存大小
    )
    
    clf.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"训练完成，耗时: {train_time:.2f} 秒")
    
    # 验证集评估
    print("\n4. 验证集评估...")
    val_start = time.time()
    val_pred = clf.predict(X_val)
    val_prob = clf.predict_proba(X_val)[:, 1]
    val_time = time.time() - val_start
    
    print(f"验证集评估完成，耗时: {val_time:.2f} 秒")
    print(f"准确率: {accuracy_score(y_val, val_pred):.4f}")
    print(f"混淆矩阵:\n{confusion_matrix(y_val, val_pred)}")
    try:
        print(f"AUC: {roc_auc_score(y_val, val_prob):.4f}")
    except:
        pass
    
    # 测试集评估
    print("\n5. 测试集评估...")
    test_start = time.time()
    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)[:, 1]
    test_time = time.time() - test_start
    
    print(f"测试集评估完成，耗时: {test_time:.2f} 秒")
    print(f"准确率: {accuracy_score(y_test, pred):.4f}")
    print(f"混淆矩阵:\n{confusion_matrix(y_test, pred)}")
    print(f"分类报告:\n{classification_report(y_test, pred)}")
    try:
        print(f"AUC: {roc_auc_score(y_test, prob):.4f}")
    except:
        pass
    
    # 输出部分样本的置信度
    print("\n6. 样本置信度分析（前10条）:")
    for i in range(min(10, len(y_test))):
        print(f"样本{i+1}: 真实标签={y_test[i]}, 预测概率={prob[i]:.4f}, 预测类别={pred[i]}")
    
    # 统计信息
    total_time = time.time() - start_time
    print(f"\n=== 训练完成 ===")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"训练时间: {train_time:.2f} 秒")
    print(f"验证时间: {val_time:.2f} 秒")
    print(f"测试时间: {test_time:.2f} 秒")

if __name__ == "__main__":
    main() 
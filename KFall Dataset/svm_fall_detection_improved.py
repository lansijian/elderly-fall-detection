import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import time
import gc
import warnings
warnings.filterwarnings('ignore')

def extract_features(df, window_size=10):
    """提取更多特征"""
    features = []
    
    # 基础传感器特征
    sensor_cols = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'EulerX', 'EulerY', 'EulerZ']
    
    for col in sensor_cols:
        # 统计特征
        features.extend([
            df[col].mean(),
            df[col].std(),
            df[col].min(),
            df[col].max(),
            df[col].median(),
            df[col].skew(),
            df[col].kurtosis()
        ])
        
        # 滑动窗口特征
        if len(df) >= window_size:
            rolling_mean = df[col].rolling(window=window_size, min_periods=1).mean()
            rolling_std = df[col].rolling(window=window_size, min_periods=1).std()
            features.extend([
                rolling_mean.mean(),
                rolling_std.mean(),
                rolling_mean.std(),
                rolling_std.std()
            ])
        else:
            features.extend([0, 0, 0, 0])
    
    # 加速度合成特征
    acc_magnitude = np.sqrt(df['AccX']**2 + df['AccY']**2 + df['AccZ']**2)
    features.extend([
        acc_magnitude.mean(),
        acc_magnitude.std(),
        acc_magnitude.max(),
        acc_magnitude.min()
    ])
    
    # 陀螺仪合成特征
    gyr_magnitude = np.sqrt(df['GyrX']**2 + df['GyrY']**2 + df['GyrZ']**2)
    features.extend([
        gyr_magnitude.mean(),
        gyr_magnitude.std(),
        gyr_magnitude.max(),
        gyr_magnitude.min()
    ])
    
    # 欧拉角合成特征
    euler_magnitude = np.sqrt(df['EulerX']**2 + df['EulerY']**2 + df['EulerZ']**2)
    features.extend([
        euler_magnitude.mean(),
        euler_magnitude.std(),
        euler_magnitude.max(),
        euler_magnitude.min()
    ])
    
    return features

def create_window_features(df, window_size=50, stride=25):
    """创建滑动窗口特征"""
    print(f"创建滑动窗口特征 (窗口大小: {window_size}, 步长: {stride})...")
    
    window_features = []
    window_labels = []
    
    # 按参与者、任务、试验分组
    for (participant, task, trial), group in df.groupby(['participant_id', 'task_id', 'trial_id']):
        group_labels = group['label'].values
        
        for i in range(0, len(group) - window_size + 1, stride):
            window_df = group.iloc[i:i + window_size]
            
            # 提取窗口特征
            features = extract_features(window_df)
            
            # 窗口标签：如果窗口内任何一帧是跌倒，则整个窗口标记为跌倒
            window_label = 1 if np.any(group_labels[i:i + window_size]) else 0
            
            window_features.append(features)
            window_labels.append(window_label)
    
    return np.array(window_features), np.array(window_labels)

def load_and_process_data():
    """加载并处理数据"""
    # 数据路径
    sitting_train = 'KFall Dataset/split_data/sitting/train.csv'
    sitting_val = 'KFall Dataset/split_data/sitting/val.csv'
    sitting_test = 'KFall Dataset/split_data/sitting/test.csv'
    walking_train = 'KFall Dataset/split_data/walking/train.csv'
    walking_val = 'KFall Dataset/split_data/walking/val.csv'
    walking_test = 'KFall Dataset/split_data/walking/test.csv'
    
    print("=== 改进版SVM跌倒检测模型训练 ===\n")
    
    # 加载数据
    print("1. 加载数据...")
    def load_and_concat(paths):
        dfs = []
        for p in tqdm(paths, desc='加载数据'):
            try:
                df = pd.read_csv(p)
                dfs.append(df)
            except Exception as e:
                print(f"加载文件 {p} 时出错: {e}")
                continue
        return pd.concat(dfs, ignore_index=True)
    
    df_train = load_and_concat([sitting_train, walking_train])
    df_val = load_and_concat([sitting_val, walking_val])
    df_test = load_and_concat([sitting_test, walking_test])
    
    print(f"原始数据大小:")
    print(f"  训练集: {len(df_train):,} 样本")
    print(f"  验证集: {len(df_val):,} 样本")
    print(f"  测试集: {len(df_test):,} 样本")
    
    # 创建滑动窗口特征
    print("\n2. 创建滑动窗口特征...")
    X_train, y_train = create_window_features(df_train)
    X_val, y_val = create_window_features(df_val)
    X_test, y_test = create_window_features(df_test)
    
    print(f"特征工程后数据大小:")
    print(f"  训练集: {len(X_train):,} 样本, {X_train.shape[1]} 特征")
    print(f"  验证集: {len(X_val):,} 样本, {X_val.shape[1]} 特征")
    print(f"  测试集: {len(X_test):,} 样本, {X_test.shape[1]} 特征")
    
    # 释放内存
    del df_train, df_val, df_test
    gc.collect()
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_models(X_train, y_train, X_val, y_val):
    """训练多个模型并比较"""
    print("\n3. 训练模型...")
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    models = {}
    results = {}
    
    # 计算类别权重
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # 1. 线性SVM
    print("训练线性SVM...")
    svm_linear = SVC(kernel='linear', C=1.0, class_weight=class_weight_dict, random_state=42, probability=True)
    svm_linear.fit(X_train_scaled, y_train)
    
    val_pred = svm_linear.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_pred)
    models['SVM_Linear'] = svm_linear
    results['SVM_Linear'] = {'accuracy': val_acc, 'scaler': scaler}
    print(f"线性SVM验证准确率: {val_acc:.4f}")
    
    # 2. RBF SVM
    print("训练RBF SVM...")
    svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight=class_weight_dict, random_state=42, probability=True)
    svm_rbf.fit(X_train_scaled, y_train)
    
    val_pred = svm_rbf.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_pred)
    models['SVM_RBF'] = svm_rbf
    results['SVM_RBF'] = {'accuracy': val_acc, 'scaler': scaler}
    print(f"RBF SVM验证准确率: {val_acc:.4f}")
    
    # 3. 随机森林
    print("训练随机森林...")
    rf = RandomForestClassifier(n_estimators=100, class_weight=class_weight_dict, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    
    val_pred = rf.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_pred)
    models['RandomForest'] = rf
    results['RandomForest'] = {'accuracy': val_acc, 'scaler': scaler}
    print(f"随机森林验证准确率: {val_acc:.4f}")
    
    return models, results

def evaluate_best_model(models, results, X_test, y_test):
    """评估最佳模型"""
    print("\n4. 评估最佳模型...")
    
    # 选择验证集表现最好的模型
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = models[best_model_name]
    best_scaler = results[best_model_name]['scaler']
    
    print(f"最佳模型: {best_model_name}")
    print(f"验证准确率: {results[best_model_name]['accuracy']:.4f}")
    
    # 测试集评估
    X_test_scaled = best_scaler.transform(X_test)
    test_pred = best_model.predict(X_test_scaled)
    test_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"\n测试集结果:")
    print(f"准确率: {accuracy_score(y_test, test_pred):.4f}")
    print(f"混淆矩阵:\n{confusion_matrix(y_test, test_pred)}")
    print(f"分类报告:\n{classification_report(y_test, test_pred)}")
    
    try:
        print(f"AUC: {roc_auc_score(y_test, test_prob):.4f}")
    except:
        pass
    
    # 输出部分样本的置信度
    print(f"\n样本置信度分析（前10条）:")
    for i in range(min(10, len(y_test))):
        print(f"样本{i+1}: 真实标签={y_test[i]}, 预测概率={test_prob[i]:.4f}, 预测类别={test_pred[i]}")
    
    return best_model_name, best_model, best_scaler

def main():
    start_time = time.time()
    
    # 加载和处理数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_process_data()
    
    # 训练模型
    models, results = train_models(X_train, y_train, X_val, y_val)
    
    # 评估最佳模型
    best_model_name, best_model, best_scaler = evaluate_best_model(models, results, X_test, y_test)
    
    # 统计信息
    total_time = time.time() - start_time
    print(f"\n=== 训练完成 ===")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"最佳模型: {best_model_name}")
    print(f"特征数量: {X_train.shape[1]}")

if __name__ == "__main__":
    main() 
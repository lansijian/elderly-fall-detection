from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import json

app = Flask(__name__)

# 全局变量
model_data = None
current_data = None
current_window_data = []  # 存储当前窗口的数据

def convert_numpy_types(obj):
    """转换NumPy类型为Python原生类型，确保JSON序列化不会出错"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    else:
        return obj

class RuleBasedModel:
    """基于规则的跌倒检测模型"""
    def __init__(self):
        self.thresholds = {
            'acc_magnitude_fall': 2.0,
            'gyr_magnitude_fall': 1.5,
            'acc_magnitude_warning': 1.5,
            'gyr_magnitude_warning': 1.0
        }
    def predict(self, X):
        return np.zeros(len(X))
    def predict_proba(self, X):
        n_samples = len(X)
        return np.column_stack([np.ones(n_samples), np.zeros(n_samples)])

class DummyScaler:
    def transform(self, X):
        return X

def load_model():
    """加载训练好的模型"""
    global model_data
    try:
        model_path = Path('fall_detection_model.pkl')
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            print("模型加载完成")
            print(f"模型信息: {model_data['training_info']}")
        else:
            print("模型文件不存在，使用简单规则判断")
            model_data = None
    except Exception as e:
        print(f"模型加载失败: {e}")
        model_data = None

def detect_motion_state(features):
    """检测运动状态"""
    # 计算合成值
    acc_magnitude = np.sqrt(features['AccX']**2 + features['AccY']**2 + features['AccZ']**2)
    gyr_magnitude = np.sqrt(features['GyrX']**2 + features['GyrY']**2 + features['GyrZ']**2)
    
    # 计算加速度变化率（简单估计）
    acc_variance = np.var([features['AccX'], features['AccY'], features['AccZ']])
    
    # 重新调整运动状态判断逻辑，考虑到静坐和站立时也会有加速度
    # 重力加速度约为1g，所以静止状态下加速度合成值应接近1
    if acc_magnitude > 3.0 or gyr_magnitude > 2.5:
        return "运动级别7", 0.9, "#dc3545"  # 红色 - 最剧烈
    elif acc_magnitude > 2.5 or gyr_magnitude > 2.0:
        return "运动级别6", 0.85, "#e83e8c"  # 粉红色 - 非常剧烈
    elif acc_magnitude > 2.0 or gyr_magnitude > 1.5:
        return "运动级别5", 0.8, "#fd7e14"  # 橙色 - 剧烈
    elif acc_magnitude > 1.7 or gyr_magnitude > 1.2:
        return "运动级别4", 0.7, "#ffc107"  # 黄色 - 中等
    elif acc_magnitude > 1.4 or gyr_magnitude > 0.9:
        return "运动级别3", 0.6, "#20c997"  # 青色 - 轻微
    elif acc_magnitude > 1.2 or gyr_magnitude > 0.6:
        return "运动级别2", 0.5, "#17a2b8"  # 蓝色 - 很轻微
    elif acc_magnitude > 1.0 or gyr_magnitude > 0.3:
        return "运动级别1", 0.4, "#6f42c1"  # 紫色 - 几乎静止
    else:
        return "静止", 0.3, "#6c757d"  # 灰色 - 完全静止

def extract_features_for_prediction(frame_data):
    """为单帧数据提取特征（使用滑动窗口）"""
    global current_window_data
    
    # 添加当前帧到窗口
    current_window_data.append(frame_data)
    
    # 保持窗口大小
    window_size = 50
    if len(current_window_data) > window_size:
        current_window_data.pop(0)
    
    # 如果窗口不够，返回None
    if len(current_window_data) < window_size:
        return None
    
    # 创建DataFrame
    df = pd.DataFrame(current_window_data)
    
    # 提取特征
    features = []
    sensor_cols = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'EulerX', 'EulerY', 'EulerZ']
    
    for col in sensor_cols:
        features.extend([
            df[col].mean(),
            df[col].std(),
            df[col].min(),
            df[col].max()
        ])
    
    # 加速度合成特征
    acc_magnitude = np.sqrt(df['AccX']**2 + df['AccY']**2 + df['AccZ']**2)
    features.extend([
        acc_magnitude.mean(),
        acc_magnitude.std(),
        acc_magnitude.max()
    ])
    
    # 陀螺仪合成特征
    gyr_magnitude = np.sqrt(df['GyrX']**2 + df['GyrY']**2 + df['GyrZ']**2)
    features.extend([
        gyr_magnitude.mean(),
        gyr_magnitude.std(),
        gyr_magnitude.max()
    ])
    
    return np.array(features)

def predict_state_with_model(features):
    """使用训练好的模型预测状态"""
    global model_data
    
    if model_data is None:
        # 使用简单规则判断
        acc_magnitude = np.sqrt(features['AccX']**2 + features['AccY']**2 + features['AccZ']**2)
        gyr_magnitude = np.sqrt(features['GyrX']**2 + features['GyrY']**2 + features['GyrZ']**2)
        
        if acc_magnitude > 2.0 or gyr_magnitude > 1.5:
            return "跌倒", 0.8
        elif acc_magnitude > 1.5:
            return "异常活动", 0.6
        else:
            return "正常", 0.9
    
    # 使用训练好的模型
    try:
        # 提取特征
        feature_vector = extract_features_for_prediction(features)
        
        if feature_vector is None:
            # 窗口数据不足，使用简单判断
            acc_magnitude = np.sqrt(features['AccX']**2 + features['AccY']**2 + features['AccZ']**2)
            if acc_magnitude > 1.5:
                return "数据不足", 0.5
            else:
                return "正常", 0.7
        
        # 标准化
        feature_vector_scaled = model_data['scaler'].transform([feature_vector])
        
        # 预测
        prediction = model_data['model'].predict(feature_vector_scaled)[0]
        probability = model_data['model'].predict_proba(feature_vector_scaled)[0]
        
        # 返回结果
        if prediction == 1:
            return "跌倒", probability[1]
        else:
            return "正常", probability[0]
            
    except Exception as e:
        print(f"模型预测失败: {e}")
        # 回退到简单规则
        acc_magnitude = np.sqrt(features['AccX']**2 + features['AccY']**2 + features['AccZ']**2)
        if acc_magnitude > 1.5:
            return "预测错误", 0.5
        else:
            return "正常", 0.7

def get_available_datasets():
    """获取可用的数据集列表"""
    datasets = []
    
    # 使用reclassified_data目录下的数据
    data_dir = Path("../KFall Dataset/reclassified_data")
    if data_dir.exists():
        # 读取主要的重新分类数据文件
        main_file = data_dir / "kfall_reclassified_data.csv"
        if main_file.exists():
            try:
                # 读取足够的数据来获取所有参与者信息
                df_sample = pd.read_csv(main_file, nrows=300000)  # 增加读取行数以确保获取所有参与者
                participants = sorted(df_sample['participant_id'].unique())
                
                # 添加显示所有参与者数据的选项
                datasets.append({
                    'name': "所有参与者 - 完整数据",
                    'path': str(main_file),
                    'participant_id': None,
                    'type': 'full_data',
                    'description': "所有参与者的完整数据集"
                })
                
                # 为每个参与者创建数据集选项
                for participant_id in participants:
                    datasets.append({
                        'name': f"参与者 {participant_id} - 完整数据",
                        'path': str(main_file),
                        'participant_id': participant_id,
                        'type': 'full_data',
                        'description': f"参与者 {participant_id} 的所有活动数据"
                    })
                
                print(f"找到 {len(participants)} 个不同的参与者: {participants}")
                
                # 添加按活动类型分类的数据集
                activity_files = {
                    'sitting': 'sitting_activities.csv',
                    'walking': 'walking_activities.csv'
                }
                
                for activity, filename in activity_files.items():
                    activity_file = data_dir / filename
                    if activity_file.exists():
                        datasets.append({
                            'name': f"{activity} 活动数据",
                            'path': str(activity_file),
                            'activity_type': activity,
                            'type': 'activity_data',
                            'description': f"所有 {activity} 活动的数据"
                        })
                        
            except Exception as e:
                print(f"读取数据集信息失败: {e}")
    
    # 添加split_data目录下的数据集（包含跌倒标签）
    split_data_dir = Path("../KFall Dataset/split_data")
    if split_data_dir.exists():
        # 添加sitting和walking的训练、验证和测试数据集
        for activity_type in ['sitting', 'walking']:
            activity_dir = split_data_dir / activity_type
            if activity_dir.exists():
                for dataset_type in ['train', 'val', 'test']:
                    dataset_file = activity_dir / f"{dataset_type}.csv"
                    if dataset_file.exists():
                        datasets.append({
                            'name': f"{activity_type} - {dataset_type} 数据集 (含跌倒标签)",
                            'path': str(dataset_file),
                            'activity_type': activity_type,
                            'type': 'split_data',
                            'dataset_type': dataset_type,
                            'description': f"{activity_type} 活动的 {dataset_type} 数据集，包含跌倒标签"
                        })
    
    return datasets

@app.route('/')
def index():
    """主页"""
    datasets = get_available_datasets()
    return render_template('index.html', datasets=datasets)

@app.route('/api/datasets')
def api_datasets():
    """获取数据集列表API"""
    datasets = get_available_datasets()
    # 转换NumPy类型为Python原生类型
    datasets = convert_numpy_types(datasets)
    return jsonify(datasets)

@app.route('/api/load_data', methods=['POST'])
def api_load_data():
    """加载指定数据集"""
    global current_data, current_window_data
    
    data = request.get_json()
    file_path = data.get('file_path')
    participant_id = data.get('participant_id')
    activity_type = data.get('activity_type')
    
    try:
        # 如果是加载所有参与者的数据，限制数据量
        if participant_id is None and activity_type is None:
            print("加载所有参与者数据，限制数据量以提高性能")
            # 使用分块读取，只读取前20万行
            df = pd.read_csv(file_path, nrows=200000)
            print(f"已读取所有参与者的前20万行数据，共 {len(df)} 帧")
        else:
            # 正常读取数据
            df = pd.read_csv(file_path)
        
        # 根据参数过滤数据
        if participant_id is not None:
            df = df[df['participant_id'] == str(participant_id)]
            print(f"过滤参与者 {participant_id} 的数据，共 {len(df)} 帧")
        
        if activity_type is not None:
            df = df[df['activity_type'] == activity_type]
            print(f"过滤活动类型 {activity_type} 的数据，共 {len(df)} 帧")
        
        # 按时间戳排序，确保时序正确
        if 'TimeStamp(s)' in df.columns:
            df = df.sort_values('TimeStamp(s)').reset_index(drop=True)
        
        # 翻译跌倒描述（如果存在）
        if 'fall_description' in df.columns:
            # 定义跌倒描述翻译映射
            fall_description_translations = {
                'fall from standing': '从站立状态跌倒',
                'fall from sitting': '从坐姿跌倒',
                'fall from walking': '从行走中跌倒',
                'fall from running': '从奔跑中跌倒',
                'fall backward': '向后跌倒',
                'fall forward': '向前跌倒',
                'fall sideways': '侧向跌倒',
                'fall from bed': '从床上跌落',
                'fall from chair': '从椅子上跌落',
                'slip and fall': '滑倒',
                'trip and fall': '绊倒',
                'sudden fall': '突然跌倒',
                'gradual fall': '渐进性跌倒',
                'syncope': '晕厥',
                'loss of balance': '失去平衡',
                'dizziness': '头晕',
                'weakness': '虚弱',
                'vertigo': '眩晕',
                'faint': '昏厥',
                'collapse': '倒塌',
                'stumble': '踉跄',
                'slow fall': '缓慢跌倒'
            }
            
            # 应用翻译
            def translate_description(desc):
                if pd.isna(desc) or desc == '':
                    return ''
                    
                # 翻译具体内容
                translated = desc
                for eng, chn in fall_description_translations.items():
                    if eng in desc.lower():
                        translated = translated.replace(eng, f"{eng} ({chn})")
                
                return translated
            
            df['fall_description'] = df['fall_description'].apply(translate_description)
            print("已翻译跌倒描述")
        
        current_data = df
        current_window_data = []  # 重置窗口数据
        
        # 返回数据基本信息
        info = {
            'total_frames': len(df),
            'columns': df.columns.tolist(),
            'sample_rate': 100,  # 假设100Hz采样率
            'duration': len(df) / 100,  # 秒
            'participant_id': participant_id,
            'activity_type': activity_type
        }
        
        # 如果有参与者ID，添加参与者特定信息
        if participant_id is not None and len(df) > 0:
            participant_data = df[df['participant_id'] == participant_id]
            info['participant_info'] = {
                'total_frames': len(participant_data),
                'activities': participant_data['activity_type'].unique().tolist(),
                'tasks': participant_data['task_id'].unique().tolist()
            }
        
        # 转换NumPy类型为Python原生类型
        info = convert_numpy_types(info)
        
        return jsonify({'success': True, 'info': info})
    except Exception as e:
        print(f"加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/frame_data/<int:frame_id>')
def api_frame_data(frame_id):
    """获取指定帧的数据"""
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data loaded'})
    
    if frame_id >= len(current_data):
        return jsonify({'error': 'Frame index out of range'})
    
    frame = current_data.iloc[frame_id]
    
    # 提取传感器数据
    sensor_data = {
        'timestamp': frame.get('TimeStamp(s)', frame_id / 100),
        'frame_id': frame_id,
        'AccX': float(frame['AccX']),
        'AccY': float(frame['AccY']),
        'AccZ': float(frame['AccZ']),
        'GyrX': float(frame['GyrX']),
        'GyrY': float(frame['GyrY']),
        'GyrZ': float(frame['GyrZ']),
        'EulerX': float(frame['EulerX']),
        'EulerY': float(frame['EulerY']),
        'EulerZ': float(frame['EulerZ']),
        'label': int(frame['label']) if 'label' in frame else (1 if frame.get('activity_type', '') == 'fall' else 0)
    }
    
    # 添加参与者信息
    if 'participant_id' in frame:
        sensor_data['participant_id'] = str(frame['participant_id'])
    if 'task_id' in frame:
        sensor_data['task_id'] = str(frame['task_id'])
    if 'trial_id' in frame:
        sensor_data['trial_id'] = str(frame['trial_id'])
    if 'activity_type' in frame:
        sensor_data['activity_type'] = frame['activity_type']
    if 'fall_description' in frame:
        sensor_data['fall_description'] = frame['fall_description']
    if 'fall_onset_frame' in frame:
        sensor_data['fall_onset_frame'] = frame['fall_onset_frame']
    if 'fall_impact_frame' in frame:
        sensor_data['fall_impact_frame'] = frame['fall_impact_frame']
    
    # 预测跌倒状态
    fall_state, fall_confidence = predict_state_with_model(sensor_data)
    sensor_data['predicted_state'] = fall_state
    sensor_data['confidence'] = fall_confidence
    
    # 检测运动状态
    motion_state, motion_confidence, motion_color = detect_motion_state(sensor_data)
    sensor_data['motion_state'] = motion_state
    sensor_data['motion_confidence'] = motion_confidence
    sensor_data['motion_color'] = motion_color
    
    # 转换NumPy类型为Python原生类型
    sensor_data = convert_numpy_types(sensor_data)
    
    return jsonify(sensor_data)

@app.route('/api/range_data/<int:start_frame>/<int:end_frame>')
def api_range_data(start_frame, end_frame):
    """获取指定范围的数据"""
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data loaded'})
    
    if start_frame >= len(current_data) or end_frame >= len(current_data):
        return jsonify({'error': 'Frame index out of range'})
    
    range_data = current_data.iloc[start_frame:end_frame+1]
    
    # 转换为JSON格式
    data_list = []
    for idx, row in range_data.iterrows():
        frame_data = {
            'timestamp': row.get('TimeStamp(s)', idx / 100),
            'frame_id': idx,
            'AccX': float(row['AccX']),
            'AccY': float(row['AccY']),
            'AccZ': float(row['AccZ']),
            'GyrX': float(row['GyrX']),
            'GyrY': float(row['GyrY']),
            'GyrZ': float(row['GyrZ']),
            'EulerX': float(row['EulerX']),
            'EulerY': float(row['EulerY']),
            'EulerZ': float(row['EulerZ']),
            'label': int(row['label']) if 'label' in row else (1 if row.get('activity_type', '') == 'fall' else 0)
        }
        
        # 添加参与者信息
        if 'participant_id' in row:
            frame_data['participant_id'] = str(row['participant_id'])
        if 'task_id' in row:
            frame_data['task_id'] = str(row['task_id'])
        if 'trial_id' in row:
            frame_data['trial_id'] = str(row['trial_id'])
        if 'activity_type' in row:
            frame_data['activity_type'] = row['activity_type']
        if 'fall_description' in row:
            frame_data['fall_description'] = row['fall_description']
        if 'fall_onset_frame' in row:
            frame_data['fall_onset_frame'] = row['fall_onset_frame']
        if 'fall_impact_frame' in row:
            frame_data['fall_impact_frame'] = row['fall_impact_frame']
        
        data_list.append(frame_data)
    
    # 转换NumPy类型为Python原生类型
    data_list = convert_numpy_types(data_list)
    
    return jsonify(data_list)

@app.route('/api/model_info')
def api_model_info():
    """获取模型信息"""
    global model_data
    
    if model_data is None:
        return jsonify({
            'model_loaded': False,
            'message': '使用简单规则判断'
        })
    else:
        model_info = {
            'model_loaded': True,
            'training_info': model_data['training_info'],
            'feature_count': len(model_data['feature_names']),
            'window_size': model_data['window_size']
        }
        # 转换NumPy类型为Python原生类型
        model_info = convert_numpy_types(model_info)
        return jsonify(model_info)

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000) 
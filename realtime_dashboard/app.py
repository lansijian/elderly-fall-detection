import os
import glob
from collections import deque
import pandas as pd
import torch
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import eventlet

# 从 models.py 导入模型注册表和常量
from models import MODEL_REGISTRY, SENSOR_COLS, WINDOW_SIZE

# --- 应用配置 ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key_2'
socketio = SocketIO(app, async_mode='eventlet')

# --- 全局变量 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
simulation_task = None
is_paused = False  # 使用简单的布尔值代替Event
current_model = None
current_model_name = ""

# --- 数据加载 ---
try:
    full_df = pd.read_csv('test_data.csv')
    print("测试数据加载成功。")
except FileNotFoundError:
    full_df = None
    print("错误: 'test_data.csv' 未找到。")

# --- 路由和API ---
@app.route('/')
def index():
    # 修正：在渲染模板时直接传递模型列表，这是最可靠的方式
    models = sorted(list(MODEL_REGISTRY.keys()))
    return render_template('index.html', models=models)

@app.route('/api/models')
def get_models():
    """该API不再被前端直接使用，但保留作为备用"""
    # 从注册表中获取模型文件名
    model_files = list(MODEL_REGISTRY.keys())
    # 将其格式化为前端期望的格式
    available_models = [{
        'id': model_file,
        'name': model_file.replace('.pth', '').replace('_', ' ').title()
    } for model_file in model_files]
    return jsonify(available_models)

# --- WebSocket 事件处理 ---
@socketio.on('connect')
def handle_connect():
    """处理客户端连接"""
    print('客户端已连接')
    if not current_model: load_model(list(MODEL_REGISTRY.keys())[0])

@socketio.on('disconnect')
def handle_disconnect():
    """处理客户端断开连接"""
    stop_simulation_globally() # 客户端断开也停止模拟
    print('客户端已断开')

@socketio.on('start_simulation')
def handle_start_simulation(data):
    """处理开始模拟事件，只有在没有任务运行时才启动"""
    global simulation_task, is_paused
    if simulation_task:
        print("请求被忽略：一个模拟已在进行中。")
        return

    model_name = data.get('model_id')
    speed = data.get('speed', 1.0)
    is_paused = False
    print(f"开始新的模拟: 模型={model_name}, 速度={speed}x")
    simulation_task = socketio.start_background_task(
        run_simulation_task, model_name, speed
    )

@socketio.on('pause_simulation')
def handle_pause_simulation():
    global is_paused
    is_paused = True
    emit('simulation_state_update', {'state': 'paused', 'message': '模拟已暂停。'})

@socketio.on('resume_simulation')
def handle_resume_simulation():
    global is_paused
    is_paused = False
    emit('simulation_state_update', {'state': 'running', 'message': '模拟已恢复。'})

@socketio.on('stop_simulation')
def handle_stop_simulation():
    """处理停止模拟事件"""
    stop_simulation_globally()

def load_model(model_name_to_load):
    """根据名称加载模型"""
    global current_model, current_model_name
    
    if model_name_to_load not in MODEL_REGISTRY:
        print(f"错误: 模型 '{model_name_to_load}' 未在注册表中定义。")
        return None
        
    model_path = model_name_to_load
    if not os.path.exists(model_path):
        print(f"错误: 模型文件未找到于 '{model_path}'")
        return None

    try:
        ModelClass = MODEL_REGISTRY[model_name_to_load]
        model = ModelClass().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        current_model = model
        current_model_name = model_name_to_load
        print(f"成功加载模型: {model_name_to_load}")
        return model
    except Exception as e:
        print(f"加载模型 {model_name_to_load} 时出错: {e}")
        return None

def run_simulation_task(model_name, speed_multiplier):
    """
    运行模拟的核心后台任务
    """
    global simulation_task, is_paused

    # 1. 确保模型已加载
    if current_model is None or current_model_name != model_name:
        model = load_model(model_name)
        if model is None:
            socketio.emit('simulation_error', {'message': f'无法加载模型: {model_name}'})
            simulation_task = None # 关键：任务失败，重置状态
            return
    
    # 2. 加载数据
    try:
        df = pd.read_csv('test_data.csv')
        socketio.emit('simulation_state_update', {'state': 'running', 'message': '模拟正在运行...', 'total_steps': len(df)})
    except FileNotFoundError:
        socketio.emit('simulation_error', {'message': 'test_data.csv 未找到!'})
        simulation_task = None # 关键：任务失败，重置状态
        return

    # 3. 开始循环
    data_window = []
    sensor_cols = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'EulerX', 'EulerY', 'EulerZ']
    info_cols = ['TimeStamp(s)', 'label', 'participant_id', 'trial_id', 'fall_description']
    
    # 添加稳定性缓冲区，防止模型在窗口刚满时就立即预测跌倒
    stable_window_count = 0
    required_stable_frames = 10  # 需要额外的10帧稳定期
    
    # 为不同模型设置不同的稳定期要求
    if "cnn_gru" in model_name.lower() or "cnn_lstm" in model_name.lower():
        required_stable_frames = 15  # CNN-GRU和CNN-LSTM模型需要更长的稳定期
    
    # 跟踪上一帧的预测结果，用于检测状态变化
    last_prediction_was_fall = False
    
    for index, row in df.iterrows():
        # 停止检查：如果全局任务句柄被外部设置为了None，则停止
        if simulation_task is None:
            print("检测到停止信号，退出模拟。")
            return # 任务结束

        while is_paused: # 可靠的暂停循环
            socketio.sleep(0.2)
            if simulation_task is None: break 

        # 核心逻辑
        data_window.append(row[sensor_cols].tolist())
        if len(data_window) > WINDOW_SIZE: 
            data_window.pop(0)
            stable_window_count += 1  # 窗口满后，每帧都增加稳定计数
        
        # 预测逻辑 - 修正：只在数据窗口满且经过稳定期后进行预测
        if len(data_window) == WINDOW_SIZE and stable_window_count >= required_stable_frames:
            with torch.no_grad():
                input_tensor = torch.FloatTensor([data_window]).to(device)
                output = current_model(input_tensor)
                
                # 根据模型类型正确处理输出
                probability = output.item()
                
                # 设置阈值和预测文本
                threshold = 0.84
                is_fall = probability > threshold
                
                # 计算正确的置信度
                if is_fall:
                    confidence = probability  # 跌倒的置信度就是原始概率
                    prediction_text = f"有跌倒风险 (置信度: {confidence:.2f})"
                    last_prediction_was_fall = True
                else:
                    confidence = 1 - probability  # 正常活动的置信度是1减去跌倒概率
                    prediction_text = f"正常活动 (置信度: {confidence:.2f})"
                    
                    # 关键修复：如果上一帧是跌倒预测，而这一帧变为正常，强制发送一个状态重置信号
                    if last_prediction_was_fall:
                        socketio.emit('status_reset', {
                            'message': '跌倒状态已结束',
                            'reset_confidence': True
                        })
                        last_prediction_was_fall = False
        else:
            # 数据窗口未满或未经过稳定期，不进行预测
            prediction_text = "数据收集中..."
            probability = 0.0
            confidence = 0.0
            last_prediction_was_fall = False

        # 发送数据
        socketio.emit('update_data', {
            'progress': (index + 1) / len(df) * 100,
            'sensors': row[sensor_cols].to_dict(),
            'info': row[info_cols].to_dict(),
            'prediction': prediction_text,
            'probability': probability,  # 原始概率值
            'confidence': confidence,    # 显示用的置信度
            'true_label': int(row['label']),
            'is_fall': last_prediction_was_fall  # 添加明确的跌倒状态标志
        })
        
        # 调整速度
        base_sleep_time = 0.04  # 比之前快5倍
        sleep_duration = base_sleep_time / speed_multiplier
        socketio.sleep(sleep_duration)

    # 4. 任务正常结束
    print("模拟完成.")
    socketio.emit('simulation_state_update', {'state': 'finished', 'message': '模拟已结束或停止。'})
    simulation_task = None # 关键：任务自然结束，重置状态
    is_paused = False

def stop_simulation_globally():
    """一个明确的、全局的停止函数"""
    global simulation_task, is_paused
    if simulation_task:
        simulation_task = None
        is_paused = False
        print("停止信号已发送。")
        # 直接在此处广播结束状态，确保UI解锁
        socketio.emit('simulation_state_update', {'state': 'finished', 'message': '模拟已被用户停止。'})

if __name__ == '__main__':
    print("启动Flask服务器...")
    print("请在浏览器中打开 http://127.0.0.1:5000")
    socketio.run(app, host='0.0.0.0', port=5000)

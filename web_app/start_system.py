#!/usr/bin/env python3
"""
跌倒检测实时可视化系统启动脚本
自动训练模型并启动Web应用
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """检查依赖是否安装"""
    print("�� 检查依赖...")
    
    # 包名映射：导入名 -> pip包名
    package_mapping = {
        'flask': 'Flask',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'sklearn': 'scikit-learn'  # sklearn是scikit-learn的导入名
    }
    
    missing_packages = []
    
    for import_name, package_name in package_mapping.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name}")
    
    if missing_packages:
        print(f"\n❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_data_files():
    """检查数据文件是否存在"""
    print("\n📁 检查数据文件...")
    
    data_dir = Path("../KFall Dataset/split_data")
    required_files = [
        "sitting/train.csv",
        "sitting/val.csv", 
        "sitting/test.csv",
        "walking/train.csv",
        "walking/val.csv",
        "walking/test.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = data_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path}")
    
    if missing_files:
        print(f"\n❌ 缺少数据文件: {', '.join(missing_files)}")
        print("请确保KFall数据集位于正确位置")
        return False
    
    return True

def train_model():
    """训练模型"""
    print("\n🤖 训练SVM模型...")
    
    model_file = Path("fall_detection_model.pkl")
    if model_file.exists():
        print("✅ 模型文件已存在，跳过训练")
        return True
    
    try:
        # 运行训练脚本
        result = subprocess.run([sys.executable, "train_and_save_model.py"], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✅ 模型训练完成")
            return True
        else:
            print(f"❌ 模型训练失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 模型训练异常: {e}")
        return False

def start_web_app():
    """启动Web应用"""
    print("\n🌐 启动Web应用...")
    
    try:
        # 启动Flask应用
        print("🚀 服务器启动中...")
        print("📱 请在浏览器中访问: http://localhost:5000")
        print("⏹️  按 Ctrl+C 停止服务器")
        
        subprocess.run([sys.executable, "app.py"], cwd=os.getcwd())
        
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

def main():
    """主函数"""
    print("=" * 50)
    print("🏥 跌倒检测实时可视化系统")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查数据文件
    if not check_data_files():
        return
    
    # 训练模型
    if not train_model():
        print("⚠️  将使用简单规则判断")
    
    # 启动Web应用
    start_web_app()

if __name__ == "__main__":
    main() 
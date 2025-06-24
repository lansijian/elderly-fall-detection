#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用time_series_splits数据集训练SimpleTM跌倒检测模型
"""

import os
import sys
import subprocess
import argparse

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用time_series_splits数据集训练SimpleTM跌倒检测模型')
    
    # 训练参数
    parser.add_argument('--window_size', type=int, default=128, 
                      help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=32, 
                      help='滑动窗口步长')
    parser.add_argument('--batch_size', type=int, default=64, 
                      help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, 
                      help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, 
                      help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                      help='权重衰减')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=256, 
                      help='模型维度')
    parser.add_argument('--dropout', type=float, default=0.2, 
                      help='Dropout率')
    parser.add_argument('--e_layers', type=int, default=1, 
                      help='编码器层数')
    parser.add_argument('--alpha', type=float, default=0.5, 
                      help='几何注意力中的平衡系数')
    parser.add_argument('--kernel_size', type=int, default=4, 
                      help='小波变换核大小')
    parser.add_argument('--m', type=int, default=2, 
                      help='小波分解级别')
    
    # 数据参数
    parser.add_argument('--splits_dir', type=str, default='../KFall Dataset/time_series_splits',
                      help='预分割数据集目录')
    
    # 输出参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints/time_series_split_model',
                      help='模型保存目录')
    parser.add_argument('--results_dir', type=str, default='./results/time_series_split_model',
                      help='结果保存目录')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 构建训练命令
    train_cmd = [
        "python", "train.py",
        "--use_splits",
        "--splits_dir", args.splits_dir,
        "--window_size", str(args.window_size),
        "--stride", str(args.stride),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--d_model", str(args.d_model),
        "--dropout", str(args.dropout),
        "--e_layers", str(args.e_layers),
        "--alpha", str(args.alpha),
        "--kernel_size", str(args.kernel_size),
        "--m", str(args.m),
        "--save_dir", args.save_dir,
        "--results_dir", args.results_dir
    ]
    
    # 执行训练命令
    print("开始训练模型...")
    print("训练命令:", " ".join(train_cmd))
    
    try:
        subprocess.run(train_cmd, check=True)
        print("模型训练完成!")
    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")
        return 1
    
    # 构建评估命令
    eval_cmd = [
        "python", "evaluate.py",
        "--use_splits",
        "--splits_dir", args.splits_dir,
        "--window_size", str(args.window_size),
        "--stride", str(args.stride),
        "--batch_size", str(args.batch_size),
        "--d_model", str(args.d_model),
        "--dropout", str(args.dropout),
        "--e_layers", str(args.e_layers),
        "--alpha", str(args.alpha),
        "--kernel_size", str(args.kernel_size),
        "--m", str(args.m),
        "--model_path", os.path.join(args.save_dir, "best_model.pth"),
        "--results_dir", os.path.join(args.results_dir, "evaluation"),
        "--save_attention"
    ]
    
    # 创建评估结果目录
    os.makedirs(os.path.join(args.results_dir, "evaluation"), exist_ok=True)
    
    # 执行评估命令
    print("\n开始评估模型...")
    print("评估命令:", " ".join(eval_cmd))
    
    try:
        subprocess.run(eval_cmd, check=True)
        print("模型评估完成!")
    except subprocess.CalledProcessError as e:
        print(f"评估失败: {e}")
        return 1
    
    print("\n训练和评估流程全部完成!")
    print(f"模型保存在: {args.save_dir}")
    print(f"结果保存在: {args.results_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
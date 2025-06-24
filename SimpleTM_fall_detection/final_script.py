import os
import argparse
import subprocess
import time
import sys
import joblib

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SimpleTM跌倒检测完整流程')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='../KFall Dataset/processed_data/kfall_processed_data.csv',
                      help='传感器数据路径')
    parser.add_argument('--use_splits', action='store_true',
                      help='是否使用预分割的数据集')
    parser.add_argument('--splits_dir', type=str, default='../KFall Dataset/time_series_splits',
                      help='预分割数据集目录')
    parser.add_argument('--window_size', type=int, default=128, 
                      help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=32, 
                      help='滑动窗口步长')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, 
                      help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, 
                      help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, 
                      help='学习率')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=256, 
                      help='模型维度')
    parser.add_argument('--dropout', type=float, default=0.2, 
                      help='Dropout率')
    parser.add_argument('--alpha', type=float, default=0.5, 
                      help='几何注意力系数')
    parser.add_argument('--kernel_size', type=int, default=4, 
                      help='小波变换核大小')
    
    # 目录参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints', 
                      help='模型保存目录')
    parser.add_argument('--results_dir', type=str, default='./results', 
                      help='结果保存目录')
    
    # 功能参数
    parser.add_argument('--skip_train', action='store_true',
                      help='跳过训练阶段')
    parser.add_argument('--skip_eval', action='store_true',
                      help='跳过评估阶段')
    parser.add_argument('--skip_predict', action='store_true',
                      help='跳过预测阶段')
    parser.add_argument('--test_file', type=str, default=None,
                      help='用于预测的测试文件，如果不提供则跳过预测')
    parser.add_argument('--update_readme', action='store_true',
                      help='是否更新README.md中的训练结果')

    return parser.parse_args()

def run_command(command, description):
    """运行命令并处理输出"""
    print(f"\n{'='*80}")
    print(f"开始{description}...")
    print(f"{'='*80}")
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时输出命令执行结果
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        
        process.wait()
        
        if process.returncode != 0:
            print(f"\n{description}失败，退出码: {process.returncode}")
            return False
        
        print(f"\n{description}完成")
        return True
    except Exception as e:
        print(f"\n{description}出错: {str(e)}")
        return False

def save_execution_summary(results_dir, phases, success_flags, start_time, args):
    """保存执行摘要"""
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    summary_file = os.path.join(results_dir, "execution_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("SimpleTM跌倒检测执行摘要\n")
        f.write("=========================\n\n")
        f.write(f"执行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {elapsed_time:.2f} 秒\n\n")
        
        f.write("执行参数:\n")
        for arg, value in vars(args).items():
            f.write(f"  {arg}: {value}\n")
        f.write("\n")
        
        f.write("执行阶段:\n")
        for i, phase in enumerate(phases):
            status = "成功" if success_flags[i] else "失败"
            f.write(f"  {phase}: {status}\n")

def update_readme(evaluation_dir):
    """更新README.md中的训练结果"""
    # 查找最新的分类报告
    classification_report = os.path.join(evaluation_dir, "classification_report.txt")
    
    if os.path.exists(classification_report):
        print("\n正在更新README.md中的训练结果...")
        
        try:
            # 调用更新脚本
            cmd = [
                "python", "./utils/update_readme.py",
                "--report_file", classification_report,
                "--readme_file", "./README.md"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True
            )
            
            for line in process.stdout:
                print(line, end='')
            
            process.wait()
            
            if process.returncode == 0:
                print("README.md已更新")
            else:
                print("更新README.md失败")
        
        except Exception as e:
            print(f"更新README.md时出错: {str(e)}")

def main():
    """主函数"""
    # 记录开始时间
    start_time = time.time()
    
    # 解析参数
    args = parse_args()
    
    # 创建必要的目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 定义执行阶段
    phases = []
    success_flags = []
    
    # 打印欢迎信息
    print("\n" + "="*50)
    print("SimpleTM跌倒检测系统 - 完整流程")
    print("="*50)
    print(f"数据路径: {args.data_path}")
    if args.use_splits:
        print(f"使用预分割数据集: {args.splits_dir}")
    print(f"模型保存目录: {args.save_dir}")
    print(f"结果保存目录: {args.results_dir}")
    print("="*50)

    # 训练阶段
    if not args.skip_train:
        phases.append("训练")
        
        train_cmd = [
            "python", "train.py",
            "--data_path", args.data_path,
            "--window_size", str(args.window_size),
            "--stride", str(args.stride),
            "--batch_size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--d_model", str(args.d_model),
            "--dropout", str(args.dropout),
            "--alpha", str(args.alpha),
            "--kernel_size", str(args.kernel_size),
            "--save_dir", args.save_dir,
            "--results_dir", args.results_dir
        ]
        
        # 如果使用预分割数据集，添加相应参数
        if args.use_splits:
            train_cmd.extend(["--use_splits", "--splits_dir", args.splits_dir])
        
        success = run_command(train_cmd, "模型训练")
        success_flags.append(success)
        
        if success:
            # 保存scaler以便预测时使用
            try:
                if args.use_splits:
                    from data_processors.data_processor import load_from_time_series_splits
                    _, _, _, scaler = load_from_time_series_splits(
                        data_dir=args.splits_dir,
                        window_size=args.window_size,
                        stride=args.stride,
                        batch_size=args.batch_size,
                        normalize=True
                    )
                else:
                    from data_processors.data_processor import prepare_data_for_simpleTM
                    _, _, _, scaler = prepare_data_for_simpleTM(
                        data_path=args.data_path,
                        window_size=args.window_size,
                        stride=args.stride,
                        batch_size=args.batch_size,
                        normalize=True
                    )
                scaler_path = os.path.join(args.save_dir, 'scaler.joblib')
                joblib.dump(scaler, scaler_path)
                print(f"标准化器已保存至 {scaler_path}")
            except Exception as e:
                print(f"保存标准化器时出错: {str(e)}")
    
    # 评估阶段
    evaluation_dir = os.path.join(args.results_dir, "evaluation")
    if not args.skip_eval and (args.skip_train or success_flags[-1]):
        phases.append("评估")
        
        # 创建评估结果目录
        os.makedirs(evaluation_dir, exist_ok=True)
        
        eval_cmd = [
            "python", "evaluate.py",
            "--data_path", args.data_path,
            "--window_size", str(args.window_size),
            "--stride", str(args.stride),
            "--batch_size", str(args.batch_size),
            "--d_model", str(args.d_model),
            "--dropout", str(args.dropout),
            "--alpha", str(args.alpha),
            "--kernel_size", str(args.kernel_size),
            "--model_path", os.path.join(args.save_dir, "best_model.pth"),
            "--results_dir", evaluation_dir,
            "--save_attention"
        ]
        
        # 如果使用预分割数据集，添加相应参数
        if args.use_splits:
            eval_cmd.extend(["--use_splits", "--splits_dir", args.splits_dir])
        
        success = run_command(eval_cmd, "模型评估")
        success_flags.append(success)
    
    # 预测阶段（如果提供了测试文件）
    if not args.skip_predict and args.test_file:
        phases.append("预测")
        
        # 创建预测结果目录
        prediction_dir = os.path.join(args.results_dir, "predictions")
        os.makedirs(prediction_dir, exist_ok=True)
        
        predict_cmd = [
            "python", "predict.py",
            "--input_file", args.test_file,
            "--window_size", str(args.window_size),
            "--stride", str(args.stride),
            "--d_model", str(args.d_model),
            "--dropout", str(args.dropout),
            "--alpha", str(args.alpha),
            "--kernel_size", str(args.kernel_size),
            "--model_path", os.path.join(args.save_dir, "best_model.pth"),
            "--scaler_path", os.path.join(args.save_dir, "scaler.joblib"),
            "--output_file", os.path.join(prediction_dir, "predictions.csv"),
            "--visualize"
        ]
        
        # 如果使用预分割数据集，添加相应参数
        if args.use_splits:
            predict_cmd.extend(["--use_splits", "--splits_dir", args.splits_dir])
        
        success = run_command(predict_cmd, "模型预测")
        success_flags.append(success)
    
    # 保存执行摘要
    save_execution_summary(args.results_dir, phases, success_flags, start_time, args)
    
    # 更新README.md（如果需要）
    if args.update_readme and os.path.exists(os.path.join(evaluation_dir, "classification_report.txt")):
        update_readme(evaluation_dir)
    
    # 创建结果摘要CSV
    try:
        run_command(
            ["python", "./utils/update_readme.py", "--results_dir", args.results_dir],
            "创建结果摘要"
        )
    except:
        print("创建结果摘要失败")
    
    # 打印执行结果摘要
    print("\n" + "="*50)
    print("执行结果摘要")
    print("="*50)
    
    for i, phase in enumerate(phases):
        status = "成功" if success_flags[i] else "失败"
        print(f"{phase}: {status}")
    
    print("\n执行完成。结果保存在: " + args.results_dir)

if __name__ == "__main__":
    main() 

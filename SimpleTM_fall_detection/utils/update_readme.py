#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
更新README.md文件的训练结果表格
读取分类报告并提取相关性能指标，然后更新README.md中的训练结果表格
"""

import re
import os
import argparse
from datetime import datetime

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='更新README.md中的训练结果')
    
    parser.add_argument('--report_file', type=str, required=True,
                      help='分类报告文件路径')
    parser.add_argument('--readme_file', type=str, default='./README.md',
                      help='README.md文件路径')
    
    return parser.parse_args()

def extract_metrics_from_report(report_file):
    """从分类报告中提取性能指标"""
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取模型配置信息
        configs = {}
        window_size_match = re.search(r'窗口大小: (\d+)', content)
        if window_size_match:
            configs['window_size'] = window_size_match.group(1)
        
        d_model_match = re.search(r'模型维度: (\d+)', content)
        if d_model_match:
            configs['d_model'] = d_model_match.group(1)
        
        e_layers_match = re.search(r'编码器层数: (\d+)', content)
        if e_layers_match:
            configs['e_layers'] = e_layers_match.group(1)
        
        alpha_match = re.search(r'几何注意力系数 \(alpha\): ([\d.]+)', content)
        if alpha_match:
            configs['alpha'] = alpha_match.group(1)
        
        # 提取性能指标
        # 注意，这里需要注意分类报告格式，通常我们使用weighted avg行
        metrics = {}
        weighted_avg_match = re.search(r'weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', content)
        if weighted_avg_match:
            metrics['precision'] = weighted_avg_match.group(1)
            metrics['recall'] = weighted_avg_match.group(2)
            metrics['f1'] = weighted_avg_match.group(3)
            metrics['support'] = weighted_avg_match.group(4)
        
        # 寻找准确度
        accuracy_match = re.search(r'accuracy\s+([\d.]+)', content)
        if accuracy_match:
            metrics['accuracy'] = accuracy_match.group(1)
        
        return configs, metrics
    
    except Exception as e:
        print(f"提取指标时出错: {e}")
        return {}, {}

def update_readme_table(readme_file, configs, metrics):
    """更新README.md中的训练结果表格"""
    try:
        with open(readme_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 寻找训练结果表格
        table_pattern = r'(\| 模型配置 \| 准确率 \| 精确率 \| 召回率 \| F1分数 \|[\s\S]*?\n)(\|---------|-------|-------|--------|--------\|[\s\S]*?)(\n\*注：.*)'
        table_match = re.search(table_pattern, content)
        
        if not table_match:
            print("在README.md中未找到训练结果表格")
            return False
        
        header = table_match.group(1)
        rows = table_match.group(2)
        footer = table_match.group(3)
        
        # 构建模型配置描述
        config_desc = f"SimpleTM (w={configs.get('window_size', '-')}, d={configs.get('d_model', '-')}, α={configs.get('alpha', '-')})"
        
        # 构建新行
        new_row = f"| {config_desc} | {metrics.get('accuracy', '-')} | {metrics.get('precision', '-')} | {metrics.get('recall', '-')} | {metrics.get('f1', '-')} |\n"
        
        # 检查是否已存在该配置的行
        if config_desc in rows:
            # 替换现有行
            rows_lines = rows.split('\n')
            updated_rows = []
            for line in rows_lines:
                if config_desc in line:
                    updated_rows.append(new_row.strip())
                else:
                    updated_rows.append(line)
            rows = '\n'.join(updated_rows)
        else:
            # 添加新行
            rows = rows + new_row
        
        # 更新时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        updated_footer = re.sub(r'\*注：训练后将自动更新此表格\*', f'*注：最后更新于 {timestamp}*', footer)
        
        # 更新内容
        updated_content = content.replace(table_match.group(0), header + rows + updated_footer)
        
        # 写入文件
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"已成功更新README.md中的训练结果表格")
        return True
    
    except Exception as e:
        print(f"更新README.md时出错: {e}")
        return False

def main():
    """主函数"""
    args = parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.report_file):
        print(f"错误: 分类报告文件 {args.report_file} 不存在!")
        return
    
    if not os.path.exists(args.readme_file):
        print(f"错误: README文件 {args.readme_file} 不存在!")
        return
    
    # 提取指标
    configs, metrics = extract_metrics_from_report(args.report_file)
    
    if not configs or not metrics:
        print("从分类报告中提取性能指标失败")
        return
    
    # 更新README表格
    success = update_readme_table(args.readme_file, configs, metrics)
    
    if success:
        print("操作完成")
    else:
        print("更新README.md失败")

if __name__ == "__main__":
    main() 
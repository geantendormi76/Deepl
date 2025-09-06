# 文件: train.py (最终版 - 适配简化命名)
import argparse
import json
from pathlib import Path
import importlib

# 【核心修正】导入语句已根据您的命名偏好简化
from trainer.trainer import Trainer

def main(config, task_name):
    """
    项目的主训练入口。
    """
    task_config = config['tasks'][task_name]
    print(f"\n--- 启动训练任务: {task_config['description']} ---")

    # 1. 动态调用数据准备模块
    print("\n[Step 1/2] 准备数据集...")
    data_builder_module_name = task_config['data_builder_module']
    data_builder_func_name = task_config['data_builder_func']
    
    try:
        module = importlib.import_module(data_builder_module_name)
        prepare_data_func = getattr(module, data_builder_func_name)
        prepare_data_func(config)
    except Exception as e:
        print(f"❌ 错误: 执行数据准备脚本时失败。")
        print(f"   - 详情: {e}")
        return

    # 2. 初始化并启动训练器
    print("\n[Step 2/2] 初始化并启动训练器...")
    yolo_config_path = task_config['yolo_config_path']
    trainer_instance = Trainer(yolo_config_path)
    trainer_instance.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MHXY AI Model Factory')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                      help='Path to the main configuration file (default: config.json)')
    parser.add_argument('-t', '--task', type=str, required=True, choices=['detector', 'classifier'],
                      help='Name of the task to run (detector or classifier)')
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"❌ 错误: 找不到配置文件 '{config_path}'")
    else:
        config = json.loads(config_path.read_text())
        main(config, args.task)

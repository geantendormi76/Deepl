# 文件: train.py (V3 - 优雅架构最终版)
import argparse
import json
from pathlib import Path

# 【核心】我们在这里定义一个项目级的约定：所有配置文件都放在 configs 目录下
CONFIG_DIR = Path("configs")

# 导入 Trainer 时不再需要复杂的 sys.path 操作，因为它应该是一个可安装的包或有正确的 __init__.py
from trainer.trainer import Trainer

def main(config, task_name):
    task_config = config['tasks'][task_name]
    print(f"\n--- 启动训练任务: {task_config['description']} ---")

    # [步骤 1/1] 初始化并启动训练器
    print("\n[Step 1/1] 初始化并启动训练器...")
    
    # 【核心修正】路径拼接的逻辑被统一收归于此，健壮且不易出错
    # 1. 从主配置中获取纯粹的文件名
    config_filename = task_config['yolo_config_path']
    # 2. 与我们约定的配置目录进行拼接
    yolo_config_path = CONFIG_DIR / config_filename

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
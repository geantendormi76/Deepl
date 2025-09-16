# 文件: /home/zhz/deepl/prepare_data.py
# 职责: 专门负责调用数据加载器，将原始数据 (如.json) 转换为YOLO格式。

import argparse
import json
from pathlib import Path
import importlib

def main(config, task_name):
    task_config = config['tasks'][task_name]
    print(f"\n--- 启动数据准备任务: {task_config['description']} ---")
    
    data_builder_module_name = task_config['data_builder_module']
    data_builder_func_name = task_config['data_builder_func']
    
    try:
        module = importlib.import_module(data_builder_module_name)
        prepare_data_func = getattr(module, data_builder_func_name)
        prepare_data_func(config)
        print(f"\n✅ 数据准备任务 '{task_name}' 完成！")
    except Exception as e:
        print(f"❌ 错误: 执行数据准备脚本时失败。详情: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MHXY AI Model Factory - Data Preparation')
    parser.add_argument('-c', '--config', default='config.json', type=str)
    parser.add_argument('-t', '--task', type=str, required=True, choices=['detector', 'classifier'])
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"❌ 错误: 找不到配置文件 '{config_path}'")
    else:
        config = json.loads(config_path.read_text())
        main(config, args.task)
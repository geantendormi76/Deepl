# 文件: /home/zhz/deepl/data_loader/main_data_loader.py
from pathlib import Path

def build_yolo_detection_dataset(data_paths: dict):
    """
    准备YOLO检测任务的数据集。
    (此处应包含原来 00_json_to_txt_sample.py 的核心逻辑)
    """
    source_dir = Path(data_paths['detection_source_dir'])
    output_dir = Path(data_paths['detection_output_dir'])
    print(f"准备检测数据集: 从 {source_dir} 到 {output_dir}")
    # ... 在这里粘贴和调整你的json_to_txt转换代码 ...
    print("检测数据集准备完成 (占位符)。")


def build_yolo_classification_dataset(data_paths: dict):
    """
    准备YOLO分类任务的数据集。
    (此处应包含原来 01_data_generator.py 的核心逻辑)
    """
    asset_dir = Path(data_paths['classification_asset_dir'])
    output_dir = Path(data_paths['classification_output_dir'])
    print(f"准备分类数据集: 从 {asset_dir} 到 {output_dir}")
    # ... 在这里粘贴和调整你的数据生成代码 ...
    print("分类数据集准备完成 (占位符)。")
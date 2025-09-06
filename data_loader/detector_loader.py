# 文件: data_loader/detection_dataset_builder.py (适配版)
import sys
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict
import yaml

# 确保能导入utils模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.constants import CLASS_TO_ID, ID_TO_CLASS

def build_yolo_dataset(config: Dict):
    """
    转换LabelMe标注为YOLOv8格式并自动划分数据集。
    :param config: 从主config.json加载的完整配置字典。
    """
    data_paths = config['data_paths']
    source_dir = Path(data_paths['detector_source_dir'])
    output_dir = Path(data_paths['detector_output_dir'])
    val_split_ratio = 0.2 # 可以硬编码或从配置中读取

    print(f"--- [数据构建] 开始处理源数据: {source_dir} ---")
    
    if not source_dir.is_dir():
        print(f"⚠️ 警告: 源数据目录 {source_dir} 不存在，跳过数据准备。")
        print("   请将LabelMe标注文件放入该目录。")
        return

    if output_dir.exists():
        print(f"清理旧数据集目录: {output_dir}")
        shutil.rmtree(output_dir)
    (output_dir / "images/train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images/val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels/train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels/val").mkdir(parents=True, exist_ok=True)

    json_files = list(source_dir.glob("*.json"))
    if not json_files:
        print(f"❌ 错误: 在 {source_dir} 中未找到任何 .json 文件。")
        return

    train_files, val_files = train_test_split(json_files, test_size=val_split_ratio, random_state=42)
    print(f"数据集划分完成: {len(train_files)} 训练, {len(val_files)} 验证。")

    for split_name, file_list in [("train", train_files), ("val", val_files)]:
        for json_path in tqdm(file_list, desc=f"转换 {split_name} 集"):
            image_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                if json_path.with_suffix(ext).exists():
                    image_path = json_path.with_suffix(ext)
                    break
            
            if image_path is None: continue
            
            shutil.copy(image_path, output_dir / f"images/{split_name}/{image_path.name}")
            
            yolo_labels = convert_single_json_to_yolo(json_path, CLASS_TO_ID)
            label_path = output_dir / f"labels/{split_name}/{json_path.stem}.txt"
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_labels))
                
    dataset_yaml_data = {
        'path': str(output_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'names': ID_TO_CLASS
    }
    dataset_yaml_path = output_dir / "dataset.yaml"
    with open(dataset_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_yaml_data, f, sort_keys=False, allow_unicode=True)
    
    print(f"✅ [数据构建] 检测数据集构建成功！\n   输出目录: {output_dir}")

def convert_single_json_to_yolo(json_path: Path, class_mapping: dict) -> List[str]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    img_h, img_w = data['imageHeight'], data['imageWidth']
    yolo_labels = []
    for shape in data['shapes']:
        label_with_prefix = shape['label']
        if '-' in label_with_prefix:
            label = label_with_prefix.split('-', 1)[1]
        else:
            label = label_with_prefix
            
        if label not in class_mapping: continue
        
        class_id = class_mapping[label]
        points = np.array(shape['points'])
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        cx = (x_min + x_max) / 2 / img_w
        cy = (y_min + y_max) / 2 / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h
        
        yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    
    return yolo_labels

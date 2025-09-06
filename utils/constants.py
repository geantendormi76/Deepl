# 文件: utils/constants.py (黄金标准最终版)
from pathlib import Path
import re

# 【核心修正】路径指向项目根目录下的 configs/ 文件夹
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELME_CONFIG_PATH = PROJECT_ROOT / "configs/labelme_config.txt"

def _load_and_parse_classes_from_labelme_config():
    """
    直接从唯一的 labelme_config.txt 读取所有类别，并移除前缀。
    """
    if not LABELME_CONFIG_PATH.exists():
        raise FileNotFoundError(f"错误: 类别配置文件未找到于 {LABELME_CONFIG_PATH}")

    class_names = []
    with open(LABELME_CONFIG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 从 "1a-大海龟" 中解析出 "大海龟"
            if '-' in line:
                name = line.split('-', 1)[1]
            else:
                name = line
            
            if name not in class_names:
                class_names.append(name)
    
    return sorted(class_names)

# --- 核心常量 ---
ENTITY_CLASSES = _load_and_parse_classes_from_labelme_config()

# --- 映射关系 ---
CLASS_TO_ID = {name: i for i, name in enumerate(ENTITY_CLASSES)}
ID_TO_CLASS = {i: name for i, name in enumerate(ENTITY_CLASSES)}
NUM_CLASSES = len(ENTITY_CLASSES)

print(f"✅ [constants.py] 成功加载 {NUM_CLASSES} 个实体类别。")

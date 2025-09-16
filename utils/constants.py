# 文件: /home/zhz/deepl/utils/constants.py (V6.0 - 权威配置最终版)
# 职责: 项目的唯一事实来源，只从 labelme_config.txt 构建类别，确保全局绝对一致。

from pathlib import Path
from typing import List, Dict

# --- 核心配置 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 【核心】现在，唯一的“真理之源”是这个您亲自维护的配置文件
LABELME_CONFIG_PATH = PROJECT_ROOT / "configs" / "labelme_config.txt"

# 【核心】我们在这里定义哪些前缀的类别是需要训练的
TRAINING_PREFIXES = ('1-unit-', '2-status-', '3-item-', '4-ui-', '5-skill-')

def _load_and_parse_classes() -> List[str]:
    """
    从权威的 labelme_config.txt 读取所有类别，并根据 TRAINING_PREFIXES 进行筛选和解析。
    【核心修正】使用列表代替集合，以保证类别顺序与文件中的定义完全一致。
    """
    if not LABELME_CONFIG_PATH.exists():
        raise FileNotFoundError(f"错误: 权威类别配置文件未找到于 {LABELME_CONFIG_PATH}")

    training_classes = []  # <--- 改为列表
    with open(LABELME_CONFIG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line.startswith(TRAINING_PREFIXES):
                specific_name = line.rsplit('-', 1)[-1]
                if specific_name not in training_classes: 
                    training_classes.append(specific_name)

    return training_classes 

# --- 执行加载并生成核心常量 ---
TRAINING_CLASSES: List[str] = _load_and_parse_classes()

# --- 映射关系 ---
CLASS_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(TRAINING_CLASSES)}
ID_TO_CLASS: Dict[int, str] = {i: name for i, name in enumerate(TRAINING_CLASSES)}
NUM_CLASSES: int = len(TRAINING_CLASSES)

# --- 启动自检与信息打印 ---
print("="*50)
print("✅ [constants.py] 模块已成功加载 (权威配置模式)。")
print(f"   - 权威配置文件: {LABELME_CONFIG_PATH}")
print(f"   - 本次模型将训练的类别总数: {NUM_CLASSES}")
if TRAINING_CLASSES:
    print(f"   - 示例类别 -> ID: '{TRAINING_CLASSES[0]}' -> {CLASS_TO_ID[TRAINING_CLASSES[0]]}")
print("="*50)
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
    """
    if not LABELME_CONFIG_PATH.exists():
        raise FileNotFoundError(f"错误: 权威类别配置文件未找到于 {LABELME_CONFIG_PATH}")

    training_classes = set()
    with open(LABELME_CONFIG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 检查这一行是否是我们想要训练的类别
            if line.startswith(TRAINING_PREFIXES):
                # 从 '1-unit-enemy-mob-大海龟' 解析出 '大海龟'
                specific_name = line.rsplit('-', 1)[-1]
                training_classes.add(specific_name)

    return sorted(list(training_classes))

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
# 文件: /home/zhz/deepl/tools/data_inspector.py
# 职责: 扫描指定的标注数据源，统计所有类别的实例数量，
#       并生成一份稀缺类别清单，用于指导后续的数据合成。

import json
from pathlib import Path
from collections import Counter
import argparse
from tqdm import tqdm

# --- [核心配置] ---
# 脚本会自动计算项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 【请确认】这里指向您整合了所有真实数据的源文件夹
DATA_SOURCE_DIR = PROJECT_ROOT / "data/raw/real_data_combined_v1"

# 【可调整】当一个类别的样本数量低于此阈值时，它将被列为“稀缺类别”
DEFAULT_SCARCITY_THRESHOLD = 20

def analyze_class_distribution(data_dir: Path, threshold: int):
    """
    分析指定目录中所有LabelMe .json文件的类别分布。
    """
    print("--- 🚀 启动数据集类别分布分析器 ---")
    if not data_dir.is_dir():
        print(f"❌ 错误: 找不到数据源目录: {data_dir}")
        return

    json_files = list(data_dir.glob("*.json"))
    if not json_files:
        print(f"❌ 错误: 在 {data_dir} 中未找到任何 .json 文件。")
        return

    print(f"🔍 正在扫描 {len(json_files)} 个标注文件...")

    class_counter = Counter()

    for json_path in tqdm(json_files, desc="分析进度"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for shape in data.get('shapes', []):
                label_with_prefix = shape.get('label', '')
                if not label_with_prefix:
                    continue
                
                # 【关键】使用与训练时完全相同的解析逻辑！
                # 从 '1a-陪练' 或 '1-unit-enemy-mob-大海龟' 中解析出核心类别名
                class_name = label_with_prefix.rsplit('-', 1)[-1]
                class_counter[class_name] += 1
        except Exception as e:
            print(f"\n⚠️ 警告: 处理文件 {json_path.name} 时出错: {e}")

    if not class_counter:
        print("❌ 分析完成，但未统计到任何类别。请检查您的.json文件内容。")
        return
        
    print("\n" + "="*80)
    print("--- ✅ 数据集类别分布分析报告 ---")
    print(f"   - 扫描目录: {data_dir}")
    print(f"   - 总计发现 {len(class_counter)} 个独立类别。")
    print("="*80)

    # 1. 打印完整的类别分布情况
    print("\n--- [完整类别分布] (按数量降序) ---")
    print(f"{'排名':<5} | {'类别名称':<20} | {'实例数量':<10}")
    print("-" * 45)
    for i, (class_name, count) in enumerate(class_counter.most_common()):
        print(f"{i+1:<5} | {class_name:<20} | {count:<10}")

    # 2. 筛选并打印稀缺类别清单
    print("\n" + "="*80)
    print(f"--- [稀缺类别清单] (样本数 < {threshold}) ---")
    print("="*80)
    
    scarce_classes = []
    for class_name, count in sorted(class_counter.items(), key=lambda item: item[1]):
        if count < threshold:
            scarce_classes.append(class_name)
            print(f"   - {class_name:<20} | 仅有 {count} 个样本")

    if not scarce_classes:
        print("🎉 恭喜！没有发现任何低于阈值的稀缺类别。")
    
    # 3. 输出最终可用的清单
    print("\n" + "="*80)
    print("--- 📋 [最终稀缺类别清单] (可直接用于指导数据合成) ---")
    print("="*80)
    print(scarce_classes)
    print("\n--- 分析完成 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析LabelMe数据集的类别分布并找出稀缺类别。")
    parser.add_argument(
        '-t', '--threshold', 
        type=int, 
        default=DEFAULT_SCARCITY_THRESHOLD,
        help=f"定义稀缺类别的数量阈值 (默认: {DEFAULT_SCARCITY_THRESHOLD})"
    )
    args = parser.parse_args()
    
    analyze_class_distribution(DATA_SOURCE_DIR, args.threshold)
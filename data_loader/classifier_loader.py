# 文件: data_loader/classification_dataset_builder.py (适配版)
import cv2
import numpy as np
from pathlib import Path
import random
import shutil
from tqdm import tqdm
import sys
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def generate_classifier_data(config: Dict):
    """
    通过增强素材，生成YOLO分类任务的数据集。
    :param config: 从主config.json加载的完整配置字典。
    """
    data_paths = config['data_paths']
    asset_dir = Path(data_paths['classifier_asset_dir'])
    output_dir = Path(data_paths['classifier_output_dir'])
    
    print(f"--- [数据构建] 开始生成分类器数据: {asset_dir} ---")

    if not asset_dir.is_dir():
        print(f"⚠️ 警告: 素材目录 {asset_dir} 不存在，跳过数据准备。")
        print("   请将分类素材放入该目录。")
        return

    # ... 此处粘贴您原来的 01_data_generator.py 的核心逻辑 ...
    # 为了保持完整性，我将整个函数逻辑复制过来
    SAMPLES_PER_CLASS = 300
    SCALE_RANGE = (0.8, 1.2)
    ROTATION_RANGE = (-15, 15)
    FLIP_CHANCE = 0.5
    
    assets = {}
    class_pinyin_names = [d.name for d in asset_dir.iterdir() if d.is_dir()]
    for pinyin_name in class_pinyin_names:
        assets[pinyin_name] = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in (asset_dir / pinyin_name).glob("*.png")]
        assets[pinyin_name] = [img for img in assets[pinyin_name] if img is not None]
    
    if not assets:
        print(f"❌ 错误: 在 {asset_dir} 中未找到任何有效的素材。")
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    for pinyin_name in assets.keys():
        (output_dir / pinyin_name).mkdir(parents=True, exist_ok=True)

    for pinyin_name, asset_list in tqdm(assets.items(), desc="生成类别"):
        if not asset_list: continue
        for i in range(SAMPLES_PER_CLASS):
            base_asset = random.choice(asset_list)
            augmented_asset = augment_asset(base_asset, SCALE_RANGE, ROTATION_RANGE, FLIP_CHANCE)
            if augmented_asset is None: continue
            final_sample = augmented_asset[:,:,:3]
            save_path = output_dir / pinyin_name / f"{pinyin_name}_{i:04d}.png"
            cv2.imwrite(str(save_path), final_sample)

    print(f"✅ [数据构建] 分类数据集构建成功！\n   输出目录: {output_dir}")

def augment_asset(asset, scale_range, rotation_range, flip_chance):
    scale = random.uniform(*scale_range)
    h, w = asset.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    if new_h <= 0 or new_w <= 0: return None
    augmented_asset = cv2.resize(asset, (new_w, new_h))
    angle = random.uniform(*rotation_range)
    center = (new_w // 2, new_h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    nW, nH = int((new_h * sin) + (new_w * cos)), int((new_h * cos) + (new_w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    augmented_asset = cv2.warpAffine(augmented_asset, M, (nW, nH))
    if random.random() < flip_chance:
        augmented_asset = cv2.flip(augmented_asset, 1)
    return augmented_asset

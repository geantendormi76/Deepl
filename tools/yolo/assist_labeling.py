# 文件: tools/yolo/assist_labeling.py
# 职责: 使用训练好的模型对新数据进行预标注，以加速人工审核。

import cv2
import json
import random
from pathlib import Path
from tqdm import tqdm
import shutil
from ultralytics import YOLO

# --- 核心配置 ---
# 【重要】这里指向您V1训练产出的最佳.pt模型
MODEL_V1_PATH = Path("/home/zhz/deepl/runs/detect/locator_v1/weights/best.pt") 

# 【重要】指向您存放新一轮原始截图的目录
NEW_RAW_IMAGES_DIR = Path("/home/zhz/deepl/data/raw/new_batch_for_v2")

# 【重要】预标注结果的输出目录
ASSISTED_OUTPUT_DIR = Path("/home/zhz/deepl/data/raw/yolo_v2_assisted")

# --- 可调参数 ---
NUM_SAMPLES = 100  # 您想标注的图片数量
CONF_THRESHOLD = 0.25 # 模型预测的置信度阈值，低于此值将被忽略

def assist_labeling():
    """
    使用V1模型对新图片进行辅助标注。
    """
    print("--- 启动模型辅助标注流程 ---")

    if not MODEL_V1_PATH.is_file():
        print(f"❌ 错误: 找不到V1模型权重文件: {MODEL_V1_PATH}")
        print("   请确认您已成功训练完V1模型。")
        return

    if not NEW_RAW_IMAGES_DIR.is_dir() or not any(NEW_RAW_IMAGES_DIR.iterdir()):
        NEW_RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"❌ 错误: 新的原始图片目录为空: {NEW_RAW_IMAGES_DIR}")
        print("   请将您想标注的100张新图片放入该目录。")
        return

    ASSISTED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载我们训练好的V1模型
    model = YOLO(MODEL_V1_PATH)
    print(f"✅ 成功加载V1模型: {MODEL_V1_PATH.name}")

    all_images = list(NEW_RAW_IMAGES_DIR.glob("*.png")) + list(NEW_RAW_IMAGES_DIR.glob("*.jpg"))
    sampled_images = random.sample(all_images, min(NUM_SAMPLES, len(all_images)))
    print(f"从 {len(all_images)} 张图片中随机抽取 {len(sampled_images)} 张进行预标注...")

    for img_path in tqdm(sampled_images, desc="模型正在辅助标注"):
        # 复制图片到目标目录
        shutil.copy(img_path, ASSISTED_OUTPUT_DIR / img_path.name)

        # 使用模型进行预测
        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)
        result = results[0] # 处理第一张图的结果
        
        height, width = result.orig_shape
        shapes = []
        for box in result.boxes:
            class_id = int(box.cls[0].cpu().numpy())
            class_name = result.names.get(class_id, f"未知ID:{class_id}")
            
            # 我们需要带前缀的标签名以便LabelMe正确加载
            # 这里我们简单地用 "1a-" 作为通用前缀，您在人工审核时可以修正
            label_with_prefix = f"1a-{class_name}"

            xyxy = box.xyxy[0].cpu().numpy()
            points = [[float(xyxy[0]), float(xyxy[1])], [float(xyxy[2]), float(xyxy[3])]]
            
            shapes.append({
                'label': label_with_prefix,
                'points': points,
                'group_id': None,
                'shape_type': 'rectangle',
                'flags': {}
            })

        # 构建LabelMe兼容的JSON数据
        labelme_data = {
            'version': "5.4.1",
            'flags': {},
            'shapes': shapes,
            'imagePath': img_path.name,
            'imageData': None,
            'imageHeight': height,
            'imageWidth': width
        }

        # 保存.json文件
        json_path = ASSISTED_OUTPUT_DIR / img_path.with_suffix('.json').name
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*50)
    print("✅ 辅助标注完成！")
    print(f"   预标注文件已生成在: {ASSISTED_OUTPUT_DIR}")
    print("   下一步: 请使用 LabelMe 打开该目录，开始您的人工审核与修正。")
    print(f"   推荐命令: labelme '{ASSISTED_OUTPUT_DIR}' --labels 'configs/labelme_config.txt'")
    print("="*50)

if __name__ == '__main__':
    assist_labeling()


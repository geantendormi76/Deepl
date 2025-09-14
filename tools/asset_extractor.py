# 文件: /home/zhz/mhxy/tools/asset_extractor.py (V4 - 数字选择最终版)
# 职责: 通过交互式点击和数字选择，稳定、高效地提取单位素材。
# 【运行环境】: Windows

import cv2
import numpy as np
from pathlib import Path
import sys

# 【核心修正】动态地将项目根目录添加到sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("错误: 未找到 'segment_anything' 库。")
    print("请在你的 Windows Python 环境中运行以下命令进行安装:")
    print("pip install opencv-python segment-anything-py torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

# 【新】导入我们升级后的常量
from services.yolo_service.config.constants import CLASSIFIER_CLASSES_CHN, CLASSIFIER_CHN_TO_PINYIN

# --- 核心配置 ---
# 【修正】使用计算出的PROJECT_ROOT，而不是依赖当前工作目录
SAM_CHECKPOINT = PROJECT_ROOT / "models" / "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
SOURCE_IMAGE_DIR = PROJECT_ROOT / "data/training_data/yolo_cls/raw_battle_screenshots" 
ASSET_OUTPUT_DIR = PROJECT_ROOT / "data/training_data/yolo_cls/assets_battle_units" 
DEVICE = "cuda"

input_point = []
input_label = []

def mouse_callback(event, x, y, flags, param):
    global input_point, input_label
    if event == cv2.EVENT_LBUTTONDOWN:
        input_point.append([x, y])
        input_label.append(1)
        print(f"  -> 已添加前景点: ({x}, {y})")

def print_usage_guide():
    print("\n" + "="*50)
    print("--- 交互式资产提取器 - 操作指南 ---")
    print("="*50)
    print("1. 【选择目标】: 在弹出的图片窗口中，用鼠标左键在你想抠图的怪物身上点击 1-2 次。")
    print("2. 【保存素材】: 对预览效果满意后，按下键盘上的 's' 键。")
    print("   - 终端会显示一个类别列表，输入对应的【数字】并按回车。")
    print("3. 【清除选择】: 按下键盘上的 'r' 键。")
    print("4. 【下一张图】: 按下键盘上的 'q' 键或 'ESC' 键。")
    print("="*50 + "\n")

def extract_assets():
    print("--- SAM 资产提取器启动 ---")
    # ... (SAM模型加载部分不变) ...
    if not SAM_CHECKPOINT.exists():
        print(f"错误: SAM模型文件未找到! 请确保它位于: {SAM_CHECKPOINT}")
        return

    print("正在加载SAM模型到GPU，首次加载可能需要一些时间...")
    try:
        sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam.to(device=DEVICE)
        predictor = SamPredictor(sam)
    except Exception as e:
        print(f"加载SAM模型失败! 错误: {e}")
        return
    print("✅ SAM模型加载完成。")
    
    print_usage_guide()

    source_files = sorted(list(SOURCE_IMAGE_DIR.glob("*.png")))
    if not source_files:
        SOURCE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"错误：请将你用来抠图的战斗截图放入以下目录后重新运行:\n{SOURCE_IMAGE_DIR}")
        return

    for source_path in source_files:
        image = cv2.imread(str(source_path))
        if image is None: continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        
        window_name = f"SAM Asset Extractor - {source_path.name} (按 'q' 退出)"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)

        global input_point, input_label
        input_point, input_label = [], []

        while True:
            vis_image = image.copy()
            for point in input_point:
                cv2.circle(vis_image, tuple(point), 5, (0, 255, 0), -1)
            
            mask = None
            if len(input_point) > 0:
                masks, scores, _ = predictor.predict(
                    point_coords=np.array(input_point),
                    point_labels=np.array(input_label),
                    multimask_output=True,
                )
                mask = masks[np.argmax(scores)]
                vis_image[mask] = vis_image[mask] * 0.5 + np.array([0, 255, 0]) * 0.5

            cv2.imshow(window_name, vis_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                if mask is not None:
                    # 【核心修正】让用户通过选择数字来指定类别
                    print("\n请为当前选择的单位指定类别:")
                    for i, name in enumerate(CLASSIFIER_CLASSES_CHN):
                        print(f"  [{i}] {name}")
                    
                    class_idx = -1
                    while class_idx < 0 or class_idx >= len(CLASSIFIER_CLASSES_CHN):
                        try:
                            class_idx_str = input(f"请输入类别对应的数字 (0-{len(CLASSIFIER_CLASSES_CHN)-1}): ")
                            class_idx = int(class_idx_str)
                        except (ValueError, IndexError):
                            print("输入无效，请输入列表中的数字。")
                    
                    chn_name = CLASSIFIER_CLASSES_CHN[class_idx]
                    pinyin_name = CLASSIFIER_CHN_TO_PINYIN[chn_name]
                    
                    class_dir = ASSET_OUTPUT_DIR / pinyin_name
                    class_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 自动生成文件名
                    existing_files = len(list(class_dir.glob("*.png")))
                    file_name = f"{pinyin_name}_{existing_files + 1:02d}.png"
                    save_path = class_dir / file_name
                    
                    # 裁剪并保存
                    y_indices, x_indices = np.where(mask)
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    b, g, r = cv2.split(image)
                    alpha = (mask * 255).astype(image.dtype)
                    rgba_asset = cv2.merge([b, g, r, alpha])
                    cropped_asset = rgba_asset[y_min:y_max+1, x_min:x_max+1]
                    
                    cv2.imwrite(str(save_path), cropped_asset)
                    print(f"✅ 已保存素材 '{chn_name}' 至: {save_path}")
                    
                    input_point, input_label = [], []
                else:
                    print("请先点击一个单位再保存！")

            elif key == ord('r'):
                input_point, input_label = [], []
                print("已重置当前点击。")

            elif key == ord('q') or key == 27:
                break
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    extract_assets()
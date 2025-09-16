# 文件: validate.py
# 职责: 加载导出的ONNX模型，对验证集中的随机图片进行推理，
#       并将检测结果（边界框和标签）可视化地绘制出来，以供人工评估。

import cv2
import numpy as np
import onnxruntime as ort
import yaml
from pathlib import Path
import random

# --- [核心配置] ---
# 指向您刚刚导出的、需要被验证的模型
MODEL_PATH = Path("saved/models/yolo_v1.onnx")

# 指向数据准备阶段生成的 dataset.yaml 文件，以找到验证集和类别名
DATASET_YAML_PATH = Path("data/processed/finetune_v2_input/dataset.yaml") 

# 在此目录中查看您模型的“答卷”
OUTPUT_DIR = Path("validation_results")

# --- [推理超参数] ---
# 模型输入的图像尺寸
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# 只有当模型的预测置信度高于此阈值时，才会被认为是有效检测
CONF_THRESHOLD = 0.1

# NMS（非极大值抑制）的阈值，用于合并重叠的检测框
IOU_THRESHOLD = 0.5


class Validator:
    """
    一个完整的ONNX模型可视化验证器。
    它封装了从预处理、推理到后处理和可视化的所有步骤。
    """
    def __init__(self, model_path, dataset_yaml_path):
        # 1. 加载ONNX模型并创建推理会话
        print(f"--- 🚀 启动YOLOv1 ONNX模型可视化验证器 ---")
        print(f"正在加载模型: {model_path}")
        self.session = ort.InferenceSession(str(model_path), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print(f"模型加载成功！使用的设备: {self.session.get_providers()[0]}")
        
        # 2. 从dataset.yaml加载类别信息和验证集路径
        print(f"正在加载数据集信息: {dataset_yaml_path}")
        with open(dataset_yaml_path, 'r', encoding='utf-8') as f:
            self.dataset_info = yaml.safe_load(f)
        self.class_names = self.dataset_info['names']
        print(f"成功加载 {len(self.class_names)} 个类别。")
        
        # 3. 准备验证图片列表
        val_images_dir = dataset_yaml_path.parent / self.dataset_info['val']
        self.val_image_paths = list(val_images_dir.glob("*.png")) + list(val_images_dir.glob("*.jpg"))
        print(f"在验证集中找到 {len(self.val_image_paths)} 张图片。")

    def run_validation(self, num_images_to_test=5):
        """
        执行验证流程。
        """
        if not self.val_image_paths:
            print("❌ 错误: 验证集中没有任何图片，无法进行验证。")
            return
            
        OUTPUT_DIR.mkdir(exist_ok=True)
        print(f"\n--- 开始随机抽取 {num_images_to_test} 张图片进行验证 ---")
        print(f"结果将保存在: {OUTPUT_DIR.resolve()}")

        # 随机选择N张图片
        selected_images = random.sample(self.val_image_paths, min(num_images_to_test, len(self.val_image_paths)))

        for i, image_path in enumerate(selected_images):
            print(f"\n[{i+1}/{num_images_to_test}] 正在处理: {image_path.name}")
            
            original_image = cv2.imread(str(image_path))
            
            # 1. 预处理
            input_tensor, scale, pad_left, pad_top = self._preprocess(original_image)
            
            # 2. 推理
            model_inputs = {self.session.get_inputs()[0].name: input_tensor}
            model_outputs = self.session.run(None, model_inputs)
            
            # 3. 后处理
            boxes, scores, class_ids = self._postprocess(model_outputs[0], scale, pad_left, pad_top, original_image.shape)
            
            # 4. 可视化
            result_image = self._draw_detections(original_image, boxes, scores, class_ids)
            
            # 5. 保存结果
            output_path = OUTPUT_DIR / f"result_{image_path.name}"
            cv2.imwrite(str(output_path), result_image)
            print(f"  -> 检测到 {len(boxes)} 个物体。结果已保存至: {output_path.name}")

        print("\n--- ✅ 验证完成！ ---")

    def _preprocess(self, image):
        """将OpenCV图像转换为模型所需的输入张量。"""
        h, w, _ = image.shape
        
        # 计算缩放比例和填充尺寸
        scale = min(INPUT_WIDTH / w, INPUT_HEIGHT / h)
        unpad_w, unpad_h = int(w * scale), int(h * scale)
        pad_w, pad_h = INPUT_WIDTH - unpad_w, INPUT_HEIGHT - unpad_h
        pad_left, pad_top = pad_w // 2, pad_h // 2
        
        # 缩放和填充
        resized_img = cv2.resize(image, (unpad_w, unpad_h))
        padded_img = cv2.copyMakeBorder(resized_img, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left, cv2.BORDER_CONSTANT)
        
        # 转换为CHW格式, 归一化, 并增加Batch维度
        input_tensor = padded_img.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, scale, pad_left, pad_top

    def _postprocess(self, output, scale, pad_left, pad_top, original_shape):
        """解码YOLO模型的原始输出，执行NMS，并将坐标映射回原始图像。"""
        # [1, 72, 8400] -> [1, 8400, 72]
        output = np.transpose(output, (0, 2, 1))[0]
        
        boxes, scores, class_ids = [], [], []
        
        for row in output:
            # [cx, cy, w, h, class_prob_0, class_prob_1, ...]
            class_probs = row[4:]
            class_id = np.argmax(class_probs)
            confidence = class_probs[class_id]
            
            if confidence > CONF_THRESHOLD:
                cx, cy, w, h = row[:4]
                
                # 将中心点坐标和宽高转换为左上角和右下角坐标
                x1 = int((cx - w / 2 - pad_left) / scale)
                y1 = int((cy - h / 2 - pad_top) / scale)
                x2 = int((cx + w / 2 - pad_left) / scale)
                y2 = int((cy + h / 2 - pad_top) / scale)
                
                boxes.append([x1, y1, x2 - x1, y2 - y1]) # NMS需要 (x, y, w, h) 格式
                scores.append(float(confidence))
                class_ids.append(class_id)
        
        # 应用非极大值抑制 (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
        
        final_boxes, final_scores, final_class_ids = [], [], []
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                final_scores.append(scores[i])
                final_class_ids.append(class_ids[i])
                
        return final_boxes, final_scores, final_class_ids

    def _draw_detections(self, image, boxes, scores, class_ids):
        """在图像上绘制检测结果。"""
        for box, score, class_id in zip(boxes, scores, class_ids):
            x, y, w, h = box
            
            # 获取类别名和颜色
            label = self.class_names.get(class_id, f"ID:{class_id}")
            color = (0, 255, 0) # 绿色
            
            # 绘制边界框
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # 准备标签文本
            text = f"{label}: {score:.2f}"
            
            # 绘制标签背景
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x, y - text_h - 5), (x + text_w, y), color, -1)
            
            # 绘制标签文字
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
        return image


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print(f"❌ 错误: 模型文件未找到: {MODEL_PATH}")
    elif not DATASET_YAML_PATH.exists():
        print(f"❌ 错误: 数据集配置文件未找到: {DATASET_YAML_PATH}")
    else:
        validator = Validator(MODEL_PATH, DATASET_YAML_PATH)
        validator.run_validation(num_images_to_test=10) # 您可以修改这里来测试更多图片
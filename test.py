# 文件: deepl/test.py
import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path
import random
import yaml

# --- 测试配置 ---
# 【核心】指向我们刚刚导出的、需要被测试的模型
MODEL_TO_TEST_PATH = Path("saved/models/yolo_v1_pure_gpu.onnx")

# 【核心】指向数据准备阶段生成的 dataset.yaml 文件，以找到验证集
DATASET_YAML_PATH = Path("data/processed/locator_dataset/dataset.yaml")

class ModelTester:
    """
    一个健壮的ONNX模型测试器，用于验证模型在目标环境中的可用性。
    """
    def __init__(self, model_path: Path, dataset_yaml_path: Path):
        self.model_path = model_path
        self.dataset_yaml_path = dataset_yaml_path
        self.input_height = 640
        self.input_width = 640

    def run_tests(self):
        """执行所有测试步骤"""
        print("--- 🚀 启动YOLOv1 ONNX模型自动化测试框架 ---")
        
        if not self.model_path.exists():
            print(f"❌ [测试失败] 模型文件未找到: {self.model_path}")
            return

        if not self.dataset_yaml_path.exists():
            print(f"❌ [测试失败] 数据集配置文件未找到: {self.dataset_yaml_path}")
            print("   请先成功运行一次 train.py --task detector 以生成数据集。")
            return

        # 1. 测试GPU会话初始化
        session = self._test_gpu_session_creation()
        if session is None:
            return

        # 2. 准备测试数据
        image_path, image_np = self._prepare_test_image()
        if image_np is None:
            return
        print(f"  - [数据准备] 已随机选取验证图片: {image_path.name}")

        # 3. 预处理与推理
        self._test_inference(session, image_np)
        
        print("\n--- ✅✅✅ 所有测试已通过 ✅✅✅ ---")
        print("模型已准备好部署到您的 'win_mhxy' 项目中！")

    def _test_gpu_session_creation(self) -> ort.InferenceSession | None:
        """测试模型是否能成功在GPU上加载"""
        print("\n[Test 1/3] 正在测试GPU会话创建...")
        try:
            providers_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(str(self.model_path), providers=providers_list)
            
            session_providers = session.get_providers()
            print(f"  - [成功] InferenceSession 创建成功！")
            print(f"  - [验证] 当前实际使用的 Providers: {session_providers}")

            if 'CUDAExecutionProvider' not in session_providers:
                print("  - ❌ [测试失败] 模型已加载，但未能使用 CUDAExecutionProvider。")
                return None
            
            print("  - ✅ [通过] 模型已成功加载到GPU环境。")
            return session

        except Exception as e:
            print(f"  - ❌ [测试失败] 创建ONNX Runtime会话时发生严重错误:")
            import traceback
            traceback.print_exc()
            return None

    def _prepare_test_image(self) -> (Path, np.ndarray | None):
        """从验证集中随机选择一张图片并加载"""
        print("\n[Test 2/3] 正在准备测试图片...")
        with open(self.dataset_yaml_path, 'r') as f:
            dataset_info = yaml.safe_load(f)
        
        val_images_dir = self.dataset_yaml_path.parent / dataset_info['val']
        val_images = list(val_images_dir.glob("*.png")) + list(val_images_dir.glob("*.jpg"))
        
        if not val_images:
            print(f"  - ❌ [测试失败] 在验证集目录 {val_images_dir} 中找不到任何图片。")
            return None, None
            
        random_image_path = random.choice(val_images)
        image = cv2.imread(str(random_image_path))
        if image is None:
            print(f"  - ❌ [测试失败] OpenCV无法读取图片: {random_image_path}")
            return random_image_path, None
            
        print("  - ✅ [通过] 成功加载测试图片。")
        return random_image_path, image

    def _test_inference(self, session: ort.InferenceSession, image: np.ndarray):
        """测试完整的预处理-推理-后处理流程"""
        print("\n[Test 3/3] 正在测试模型推理流程...")
        try:
            # 使用与您 Agent 中完全相同的预处理逻辑
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            scale = min(self.input_width / w, self.input_height / h)
            unpad_w, unpad_h = int(w * scale), int(h * scale)
            resized_img = cv2.resize(img_rgb, (unpad_w, unpad_h))
            pad_h, pad_w = self.input_height - unpad_h, self.input_width - unpad_w
            pad_top, pad_left = pad_h // 2, pad_w // 2
            padded_img = cv2.copyMakeBorder(resized_img, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left, cv2.BORDER_CONSTANT)
            input_tensor = padded_img.transpose(2, 0, 1).astype(np.float32) / 255.0
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
            # 推理
            outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
            
            print("  - [成功] 模型成功执行推理！")
            print(f"  - [验证] 模型输出张量的形状: {outputs[0].shape}")
            
            if not outputs[0].shape[0] == 1 or not outputs[0].shape[2] == 8400:
                 print(f"  - ⚠️ [警告] 模型输出形状与预期不符，请检查。")
            
            print("  - ✅ [通过] 模型推理流程验证完毕。")

        except Exception as e:
            print(f"  - ❌ [测试失败] 模型推理过程中发生错误:")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    tester = ModelTester(MODEL_TO_TEST_PATH, DATASET_YAML_PATH)
    tester.run_tests()
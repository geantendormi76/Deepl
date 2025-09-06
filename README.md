
# AI-Model-Factory: AI模型工厂

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-YOLOv8-orange)](https://ultralytics.com/)

**MHXY-AI-Model-Factory** 是一个遵循专业软件工程实践的、可复现、可扩展的深度学习模型训练框架。它旨在为AI Agent项目提供一个稳定、高效的模型生产流水线，同时也为一个通用的、可快速迁移至其他AI应用场景的模型训练“工厂”。

---

## ✨ 核心特性

-   **🚀 配置驱动**: 实验参数与代码完全分离，通过修改`.json`和`.yaml`文件即可管理所有训练任务，无需改动核心代码。
-   **📦 模块化设计**: 高内聚、低耦合的目录结构，将数据处理、模型训练、配置管理等职责清晰分离，易于维护和扩展。
-   **🔧 环境可复现**: 通过 `requirements.txt` 精确锁定依赖，确保在任何机器上都能一键构建出完全一致的训练环境。
-   **⚙️ 自动化流水线**: 从数据预处理到模型训练，再到最终导出为可部署的ONNX格式，整个流程高度自动化。
-   **💡 模型迭代闭环**: 内置“数据飞轮”工作流，支持使用旧模型辅助标注新数据，再用新数据训练新模型，实现模型的持续迭代优化。

---

## 📖 目录结构

```
/deepl/
├── configs/              # 【中央控制室】所有配置文件
├── data/                 # 【原材料仓库】存放所有原始数据集
├── models/               # 【军火库】存放从外部下载的预训练模型
├── saved/                # 【产出物仓库】存放所有训练产出物 (最终ONNX模型)
├── runs/                 # 【实验记录本】Ultralytics自动生成的详细训练日志
├── tools/                # 【开发者工具箱】辅助开发的脚本 (如辅助标注)
├── trainer/              # 【训练车间】封装核心训练与导出逻辑
├── utils/                # 【通用工具箱】存放项目通用的辅助函数
├── config.json           # 【工厂总蓝图】定义所有任务和核心路径
├── requirements.txt      # 【安装说明书】项目的Python依赖
└── train.py              # 【工厂总开关】启动所有训练任务的唯一入口
```

---

## 🚀 快速上手 (Quick Start)

### 1. 环境准备

**前提:** 您的系统中已安装 [Miniconda/Anaconda](https://www.anaconda.com/download) 和 NVIDIA GPU 驱动。

```bash
# 1. 克隆本项目 (如果尚未克隆)
# git clone https://github.com/your-username/deepl.git
# cd deepl

# 2. 创建并激活Conda虚拟环境
conda create -n ag python=3.10 -y
conda activate ag

# 3. 使用项目的“安装说明书”一键安装所有依赖
pip install -r requirements.txt
```

### 2. 准备数据和预训练模型

1.  **准备数据集:**
    *   将用于**目标检测**的LabelMe标注数据 (`.json`和图片) 放入 `data/raw/yolo_v1/` 目录。
    *   将用于**图像分类**的原始素材 (按类别分的文件夹) 放入 `data/raw/assets_battle_units/` 目录。

2.  **准备预训练模型:**
    *   从 [Ultralytics官方](https://github.com/ultralytics/ultralytics) 下载所需的预训练模型 (如 `yolo11n.pt`, `yolo11n-cls.pt`)。
    *   将它们放入 `models/` 目录。

### 3. 开始你的第一次训练！

检查并按需修改 `configs/yolo/main_config.yaml` 文件中的参数（如`epochs`, `batch`等），然后执行：

```bash
# 训练一个目标检测器
python train.py --task detector
```

训练完成后，您可以在 `saved/models/` 目录下找到可直接部署的 `yolo_v1_pure_gpu.onnx` 模型。

---

## 🛠️ 详细工作流 (Workflow)

本框架的核心是通过 `train.py` 结合 `--task` 参数来驱动不同的工作流。

### 训练目标检测器 (`--task detector`)

1.  **数据准备:** 将您的LabelMe标注数据放入 `data/raw/yolo_v1/`。
2.  **配置检查:** 打开 `configs/yolo/main_config.yaml`，确保 `model` 指向正确的 `.pt` 文件，并根据需要调整超参数。
3.  **启动训练:**
    ```bash
    python train.py --task detector
    ```
4.  **获取产出:**
    *   **可部署模型:** `saved/models/yolo_v1_pure_gpu.onnx`
    *   **详细日志:** `runs/detect/<your_run_name>/`

### 训练图像分类器 (`--task classifier`)

1.  **数据准备:** 将您的分类数据集（每个子文件夹代表一个类别）放入 `data/raw/assets_battle_units/`。
2.  **配置检查:** 打开 `configs/yolo_cls/battle_unit_classifier_config.yaml`。
3.  **启动训练:**
    ```bash
    python train.py --task classifier
    ```
4.  **获取产出:**
    *   **可部署模型:** `saved/models/guaiwu_classifier.onnx`
    *   **详细日志:** `runs/classify/<your_run_name>/`

### 模型迭代：用AI加速标注 (数据飞轮)

当您想用V1模型来标注新数据以训练V2模型时，请遵循以下流程：

1.  **准备新图片:** 将一批**未标注**的图片放入 `data/raw/new_batch_for_v2/`。
2.  **配置辅助标注脚本:** 打开 `tools/yolo/assist_labeling.py`，确保 `MODEL_V1_PATH` 指向您上一轮训练产出的最佳 `.pt` 模型 (位于`runs/detect/.../weights/best.pt`)。
3.  **运行辅助标注:**
    ```bash
    python tools/yolo/assist_labeling.py
    ```
4.  **人工审核:** 使用LabelMe打开 `data/raw/yolo_v2_assisted/` 目录，对预标注结果进行修正。
5.  **合并数据:** 将审核好的新数据从 `.../yolo_v2_assisted/` **移动**到 `.../yolo_v1/` 与旧数据合并。
6.  **更新配置:** 打开 `configs/yolo/main_config.yaml`，将 `model:` 指向V1的 `best.pt` 权重，并将 `name:` 改为新名称 (如 `locator_v2`)，可适当降低学习率 `lr0`。
7.  **启动V2训练:**
    ```bash
    python train.py --task detector
    ```
    一个新的、性能更强的ONNX模型将会生成在 `saved/models/` 目录下。

---

## 🤝 如何贡献 (Contributing)

我们欢迎任何形式的贡献！如果您有任何问题、建议或想提交代码，请遵循以下步骤：

1.  **Fork** 本仓库。
2.  创建一个新的分支 (`git checkout -b feature/AmazingFeature`)。
3.  提交您的更改 (`git commit -m 'Add some AmazingFeature'`)。
4.  将您的分支推送到远程 (`git push origin feature/AmazingFeature`)。
5.  **提交一个 Pull Request**。

请确保您的代码遵循现有的风格，并为新功能添加适当的文档。

---

## 📄 许可证 (License)

本项目采用 [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0) 许可证。详情请见 `LICENSE` 文件。


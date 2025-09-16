# AI-Model-Factory: AI模型工厂 

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-yolo11-orange)](https://ultralytics.com/)

**AI-Model-Factory** 是一个遵循专业软件工程实践的、可复现、可扩展的深度学习模型训练框架。它旨在为AI Agent项目提供一个稳定、高效的模型生产流水线，同时也为一个通用的、可快速迁移至其他AI应用场景的模型训练“工厂”。

**V2.0 核心升级**: 集成了强大的**程序化数据合成**流水线与**合成-真实数据微调 (Finetuning)**工作流，能够以极低的成本，快速生产出具备强大泛化能力的商业级模型。

---

## ✨ 核心特性

-   **🚀 配置驱动**: 实验参数与代码完全分离，通过修改`.json`和`.yaml`文件即可管理所有训练任务。
-   **📦 模块化设计**: 高内聚、低耦合的目录结构，将**数据合成**、**数据准备**、**模型训练**等职责清晰分离，易于维护和扩展。
-   **🔧 环境可复现**: 通过 `requirements.txt` 精确锁定依赖，确保在任何机器上都能一键构建出完全一致的训练环境。
-   **⚙️ 自动化流水线**: 从资产提取、数据合成、数据转换，到模型训练，再到最终导出为ONNX格式，整个流程高度自动化。
-   **💡 数据飞轮闭环**: 内置最先进的“合成数据冷启动 -> 真实数据微调”工作流，实现模型的持续、高效迭代优化。

---

## 📖 目录结构

```
/deepl/
├── configs/              # 【中央控制室】所有配置文件 (YOLO指令, 项目蓝图)
├── data/                 # 【原材料与半成品仓库】
│   ├── assets/           #   - 资产库 (前景、背景)
│   ├── raw/              #   - 原始数据 (人工标注的.json)
│   └── processed/        #   - 处理后的数据集 (YOLO格式)
├── models/               # 【军火库】存放从外部下载的预训练模型
├── runs/                 # 【实验记录本】Ultralytics自动生成的详细训练日志
├── saved/                # 【产出物仓库】存放所有训练产出物 (最终ONNX模型)
├── tools/                # 【开发者工具箱】
│   ├── synthetic/        #   - 数据合成器
│   └── yolo/             #   - (已废弃) 旧的辅助标注工具
├── trainer/              # 【训练车间】封装核心训练与导出逻辑
├── utils/                # 【通用工具箱】存放项目通用的辅助代码 (如常量)
├── config.json           # 【工厂总蓝图】定义所有任务和数据流向
├── requirements.txt      # 【安装说明书】项目的Python依赖
├── data_synthesizer.py   # 【快捷方式】直接运行数据合成
├── prepare_data.py       # 【快捷方式】直接运行人工数据的预处理
└── train.py              # 【工厂总开关】启动所有训练任务的唯一入口
```

---

## 🚀 快速上手 (Quick Start)

### 1. 环境准备

**前提:** 您的系统中已安装 [Miniconda/Anaconda](https://www.anaconda.com/download) 和 NVIDIA GPU 驱动。

```bash
# 1. 克隆本项目
# cd deepl

# 2. 创建并激活Conda虚拟环境
conda create -n deepl python=3.10 -y
conda activate deepl

# 3. 一键安装所有依赖
pip install -r requirements.txt
```

### 2. 下载预训练模型

-   从 [Ultralytics官方](https://github.com/ultralytics/ultralytics) 下载所需的预训练模型 (如 `yolo11n.pt`)。
-   将它们放入 `models/` 目录。

---

## 🛠️ 核心工作流 (Workflows)

本工厂支持三种核心的、相互独立的训练工作流。

### **工作流 A：从合成数据，从零训练 (推荐的冷启动方式)**

此流程用于在没有任何人工标注的情况下，快速训练出一个知识渊博的V1基础模型。

**1. 准备资产:**
   - **前景**: 将您用SAM等工具抠好的、带透明通道的 `.png` 素材，按照 `[前缀]-[类别]-[名称]` 的格式，分类放入 `data/assets/foregrounds/` 的子文件夹中。
   - **背景**: 将干净的背景图片放入 `data/assets/backgrounds/combat/` 和 `.../ui/`。

**2. 运行数据合成:**
   - 打开 `configs/synthesis_config.yaml` 检查合成参数。
   - 运行合成脚本。它会自动扫描您的资产，并生成海量带标签的训练数据。
     ```bash
     python tools/synthetic/data_synthesizer.py
     ```

**3. 配置并开始训练:**
   - 打开 `configs/yolo/main_config.yaml`。
   - **`model`**: 指向官方预训练模型 (如 `models/yolo11n.pt`)。
   - **`data`**: 确认指向合成数据的输出路径 (`data/processed/synthetic_locator_dataset/dataset.yaml`)。
   - **`name`**: 给您的模型起个名字 (如 `locator_v1_synthetic`)。
   - **运行训练:**
     ```bash
     python train.py --task detector
     ```

**4. 获取产出:**
   - 训练完成后，您将在 `saved/models/` 目录下找到可部署的ONNX模型。

---

### **工作流 B：从人工标注数据，从零训练**

此流程用于传统的、完全依赖人工标注数据的模型训练。

**1. 准备数据:**
   - 将您用LabelMe标注好的 `图片` 和 `.json` 文件，放入 `data/raw/` 下的一个新目录，例如 `yolo_v1_manual`。

**2. 配置数据准备任务:**
   - 打开 `config.json`。
   - **`detector_source_dir`**: 指向 `data/raw/yolo_v1_manual`。
   - **`detector_output_dir`**: 指向一个新的输出目录，例如 `data/processed/manual_dataset_v1`。

**3. 运行数据准备:**
   ```bash
   python prepare_data.py --task detector
   ```

**4. 配置并开始训练:**
   - 打开 `configs/yolo/main_config.yaml`。
   - **`model`**: 指向官方预训练模型 (如 `models/yolo11n.pt`)。
   - **`data`**: 指向您刚刚准备好的数据集 (`data/processed/manual_dataset_v1/dataset.yaml`)。
   - **`name`**: 给模型起个名字 (如 `locator_v1_from_manual`)。
   - **运行训练:**
     ```bash
     python train.py --task detector
     ```

---

### **工作流 C：微调 (Finetuning) - 融合合成与真实数据 (推荐的迭代方式)**

这是最高效的工作流。它使用少量高质量的真实数据，去“特训”一个已经见过海量合成数据的基础模型，以获得最佳的泛化能力。

**前提**: 您已经通过**工作流A**训练出了一个V1合成模型。

**1. 准备“教材” (少量高质量的人工修正数据):**
   - 使用V1模型在Windows上进行**辅助标注**，然后**人工修正**。
   - 将修正好的 `图片` 和 `.json` 文件（例如20-100份），传回到WSL的 `data/raw/` 下的新目录，例如 `corrected_batch_01`。

**2. 配置并运行数据准备:**
   - 打开 `config.json`，将 `detector_source_dir` 指向 `data/raw/corrected_batch_01`，`detector_output_dir` 指向 `data/processed/finetune_v2_input`。
   - 运行 `python prepare_data.py --task detector`。

**3. 配置微调任务:**
   - 打开 `configs/yolo/main_config.yaml`。
   - **`model`**: **【关键】** 指向您V1合成模型的最佳权重 (`runs/detect/locator_v1_synthetic/weights/best.pt`)。
   - **`data`**: 指向您的微调数据集 (`data/processed/finetune_v2_input/dataset.yaml`)。
   - **`epochs`**: 减少轮数 (如 `30-50`)，因为微调不需要从头学。
   - **`name`**: 给您的V2模型起个新名字 (如 `locator_v2_finetuned`)。

**4. 启动微调:**
   ```bash
   python train.py --task detector
   ```

**5. 迭代:**
   - 导出V2模型，部署到Windows，标注下一批数据，传回WSL，微调V3模型... 如此循环，您的模型将越来越强大。

---

## 📄 许可证 (License)

本项目采用 [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0) 许可证。

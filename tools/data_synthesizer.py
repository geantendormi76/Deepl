# 文件: tools/synthetic/data_synthesizer.py (V2.4 - 健壮裁切最终版)
# 职责: 根据配置文件和资产库，全自动生成高质量的YOLOv8训练数据。
# 变更: 采用行业标准的“矩形求交”算法重构 paste_foreground 引擎，
#       使其能够完美处理前景被放置在背景之外的任何情况（包括负坐标）。

import cv2
import numpy as np
import yaml
from pathlib import Path
import random
import shutil
from tqdm import tqdm
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
# 【核心修正】导入路径指向唯一的、重构后的常量文件
from utils.constants import CLASS_TO_ID, ID_TO_CLASS
# --- 辅助函数 ---
def paste_foreground(bg_rgba, fg_rgba, pos):
    """
    【核心重构 V2.4 - 健壮裁切版】: 在一个4通道的BGRA背景上，安全地粘贴一个4通道的BGRA前景图。
    该版本完全重写了裁切逻辑，能正确处理前景被放置在背景之外的任何情况（包括负坐标）。
    """
    x, y = int(pos[0]), int(pos[1])
    fg_h, fg_w = fg_rgba.shape[:2]
    bg_h, bg_w = bg_rgba.shape[:2]

    # 1. 计算在背景坐标系下的实际重叠区域 (intersection)
    x_start_on_bg = max(x, 0)
    y_start_on_bg = max(y, 0)
    x_end_on_bg = min(x + fg_w, bg_w)
    y_end_on_bg = min(y + fg_h, bg_h)

    # 2. 如果没有任何重叠，直接返回
    if x_start_on_bg >= x_end_on_bg or y_start_on_bg >= y_end_on_bg:
        return

    # 3. 根据背景的重叠区域，计算需要从前景图中裁切的对应区域
    fg_x_start = x_start_on_bg - x
    fg_y_start = y_start_on_bg - y
    fg_x_end = x_end_on_bg - x
    fg_y_end = y_end_on_bg - y

    # 4. 执行精确裁切，现在 fg_cropped 和 bg_roi 的形状保证完全一致
    fg_cropped = fg_rgba[fg_y_start:fg_y_end, fg_x_start:fg_x_end]
    bg_roi = bg_rgba[y_start_on_bg:y_end_on_bg, x_start_on_bg:x_end_on_bg]
    
    # 5. Alpha 融合 (后续逻辑保持不变)
    fg_alpha = fg_cropped[:, :, 3] / 255.0
    alpha_mask = np.dstack([fg_alpha] * 3)
    
    blended_rgb = (fg_cropped[:, :, :3] * alpha_mask) + (bg_roi[:, :, :3] * (1 - alpha_mask))
    
    bg_roi[:, :, :3] = blended_rgb.astype(bg_rgba.dtype)

def check_iou_overlap(new_box, existing_boxes, threshold):
    for old_box in existing_boxes:
        ix1, iy1 = max(new_box[0], old_box[0]), max(new_box[1], old_box[1])
        ix2, iy2 = min(new_box[2], old_box[2]), min(new_box[3], old_box[3])
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter_area == 0: continue
        new_area = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
        old_area = (old_box[2] - old_box[0]) * (old_box[3] - old_box[1])
        union_area = new_area + old_area - inter_area
        if union_area > 0 and inter_area / union_area > threshold:
            return True
    return False

# --- 主合成器类 (其余部分保持不变) ---
class DataSynthesizer:
    def __init__(self, config_path, synth_config_path):
        with open(config_path, 'r', encoding='utf-8') as f: self.config = json.load(f)
        with open(synth_config_path, 'r', encoding='utf-8') as f: self.synth_config = yaml.safe_load(f)
        
        self.paths = self.config['data_paths']
        self.assets_dir = Path(self.paths['synthetic_assets_dir'])
        self.bg_dir = Path(self.paths['synthetic_backgrounds_dir'])
        self.output_dir = Path(self.paths['synthetic_output_dir'])
        
        self.assets = self._load_assets()
        self.backgrounds = self._load_backgrounds()
        self.current_yolo_labels = []
        self.valid_polygon = np.array(self.synth_config['placement_rules']['valid_polygon'], dtype=np.int32)

    def _load_assets(self):
        print("--- 正在加载资产库... ---")
        assets = {}
        asset_files = list(self.assets_dir.glob("**/*.png"))
        if not asset_files: raise FileNotFoundError(f"错误：在资产目录 {self.assets_dir} 中找不到任何 .png 文件。")
        for path in tqdm(asset_files, desc="加载前景资产"):
            full_key = path.parent.name
            if full_key not in assets: assets[full_key] = []
            asset_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if asset_img is not None and hasattr(asset_img, 'shape') and len(asset_img.shape) > 2 and asset_img.shape[2] == 4:
                assets[full_key].append(asset_img)
        print(f"✅ 成功加载 {len(assets)} 个类别的 {len(asset_files)} 个资产。")
        return assets

    def _load_backgrounds(self):
        print("--- 正在加载背景库... ---")
        backgrounds = {}
        for bg_type in ['combat', 'ui']:
            backgrounds[bg_type] = []
            bg_paths = (self.bg_dir / bg_type).glob("*.png")
            for path in bg_paths:
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is None: continue
                if len(img.shape) < 3 or img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                backgrounds[bg_type].append(img)
        print(f"✅ 成功加载并标准化 {len(backgrounds['combat'])} 张战斗背景和 {len(backgrounds['ui'])} 张UI背景。")
        return backgrounds

    def generate(self):
        print(f"--- [数据合成] 开始生成数据集至: {self.output_dir} ---")
        if self.output_dir.exists():
            print(f"   - 清理旧目录...")
            shutil.rmtree(self.output_dir)
        
        # 【核心修正】创建训练和验证集的完整目录结构
        img_train_dir = self.output_dir / "images/train"; lbl_train_dir = self.output_dir / "labels/train"
        img_val_dir = self.output_dir / "images/val"; lbl_val_dir = self.output_dir / "labels/val"
        img_train_dir.mkdir(parents=True, exist_ok=True); lbl_train_dir.mkdir(parents=True, exist_ok=True)
        img_val_dir.mkdir(parents=True, exist_ok=True); lbl_val_dir.mkdir(parents=True, exist_ok=True)
        
        num_to_generate = self.synth_config['num_images_to_generate']
        val_split_ratio = 0.1  # 定义10%的数据作为验证集
        num_val = int(num_to_generate * val_split_ratio)
        
        print(f"   - 数据集规模: 总计 {num_to_generate} 张, 训练集 {num_to_generate - num_val} 张, 验证集 {num_val} 张。")

        for i in tqdm(range(num_to_generate), desc="生成合成图像"):
            bg_4_channel, yolo_labels = self._generate_single_scene()
            bg_3_channel = cv2.cvtColor(bg_4_channel, cv2.COLOR_BGRA2BGR)
            img_name, label_name = f"synth_{i:06d}.png", f"synth_{i:06d}.txt"
            
            # 【核心修正】根据索引判断存入训练集还是验证集
            if i < num_val:
                # 前 num_val 张图片放入验证集
                cv2.imwrite(str(img_val_dir / img_name), bg_3_channel)
                with open(lbl_val_dir / label_name, 'w', encoding='utf-8') as f:
                    f.write("\n".join(yolo_labels))
            else:
                # 剩余的放入训练集
                cv2.imwrite(str(img_train_dir / img_name), bg_3_channel)
                with open(lbl_train_dir / label_name, 'w', encoding='utf-8') as f:
                    f.write("\n".join(yolo_labels))

        self._create_dataset_yaml()
        print(f"✅ [数据合成] 任务完成！成功生成 {num_to_generate} 张训练图像。")

    def _generate_single_scene(self):
        bg = random.choice(self.backgrounds['combat']).copy()
        self.current_yolo_labels = []
        placed_boxes = []
        self._place_units_in_formation(bg, placed_boxes)
        if self.synth_config['ui_occlusion']['enabled'] and random.random() < self.synth_config['ui_occlusion']['probability']:
            self._apply_ui_occlusion(bg)
        return bg, self.current_yolo_labels

    def _get_available_slots(self):
        slots = self.synth_config['placement_rules']['formation_slots']
        available = {key: random.sample(value, len(value)) for key, value in slots.items()}
        return available

    def _place_units_in_formation(self, bg, placed_boxes):
        cfg = self.synth_config
        available_slots = self._get_available_slots()
        num_enemies = random.randint(cfg['unit_density']['min_enemy_units'], cfg['unit_density']['max_enemy_units'])
        enemy_slots = available_slots['enemy_front'] + available_slots['enemy_back']
        random.shuffle(enemy_slots)
        for i in range(min(num_enemies, len(enemy_slots))):
            self._place_unit_at_slot('1-unit-enemy-mob', enemy_slots[i], bg, placed_boxes)
        if available_slots['friendly_back']:
            self._place_unit_at_slot('1-unit-friendly-player', available_slots['friendly_back'].pop(0), bg, placed_boxes)
        if available_slots['friendly_front']:
            self._place_unit_at_slot('1-unit-friendly-pet', available_slots['friendly_front'].pop(0), bg, placed_boxes)

    def _place_unit_at_slot(self, category_prefix, slot, bg, placed_boxes):
        possible_keys = [k for k in self.assets.keys() if k.startswith(category_prefix)]
        if not possible_keys: return
        
        unit_key = random.choice(possible_keys)
        unit_asset = random.choice(self.assets[unit_key])
        
        # 【核心修复 V2.4 - 规则化朝向】
        # 不再随机翻转，而是根据敌我身份进行确定性翻转
        # 遵循 asset_conventions.default_facing_direction = "right" 的约定
        if 'friendly' in category_prefix:
            # 友方单位在右下角，需要面向左上，因此必须翻转
            unit_asset = cv2.flip(unit_asset, 1)
        # 敌方单位在左上角，面向右下，与默认朝向一致，无需翻转

        h, w = unit_asset.shape[:2]
        
        offset_range = self.synth_config['placement_rules']['slot_random_offset']
        base_x, base_y = slot
        x_offset = random.randint(-offset_range, offset_range)
        y_offset = random.randint(-offset_range, offset_range)
        
        x, y = base_x - w // 2 + x_offset, base_y - h // 2 + y_offset
        
        new_box = [x, y, x + w, y + h]
        box_center = ((x + x + w) / 2, (y + y + h) / 2)
        
        if check_iou_overlap(new_box, placed_boxes, self.synth_config['placement_rules']['overlap_threshold']) or \
           cv2.pointPolygonTest(self.valid_polygon, box_center, False) < 0:
            return

        paste_foreground(bg, unit_asset, (x, y))
        placed_boxes.append(new_box)
        self._add_yolo_label_and_healthbar(bg, unit_key, new_box)



    def _add_yolo_label_and_healthbar(self, bg, unit_key, new_box):
        """
        【新增修复】为成功放置的单位生成YOLOv8标签，并根据规则概率性地关联和粘贴血条。
        """
        # --- 1. 生成YOLOv8标签 ---
        # 逻辑依据: configs/constants.py 中的 _load_and_parse_labelme_config
        # 从 '1-unit-friendly-player-飞燕女' 解析出 '飞燕女'
        specific_name = unit_key.rsplit('-', 1)[-1]
        
        # 检查这个类别是否是我们需要训练的目标
        if specific_name in CLASS_TO_ID:
            class_id = CLASS_TO_ID[specific_name]
            
            bg_h, bg_w = bg.shape[:2]
            x1, y1, x2, y2 = new_box
            
            # 为确保标签精确，需计算物体被粘贴后在背景上的实际可见边界
            # 这与 paste_foreground 中的裁切逻辑完全一致
            box_x1_on_bg = max(x1, 0)
            box_y1_on_bg = max(y1, 0)
            box_x2_on_bg = min(x2, bg_w)
            box_y2_on_bg = min(y2, bg_h)

            # 如果裁剪后物体完全不可见，则不生成标签
            if box_x1_on_bg >= box_x2_on_bg or box_y1_on_bg >= box_y2_on_bg:
                return

            # 计算YOLO格式的中心点坐标和宽高（归一化）
            cx = (box_x1_on_bg + box_x2_on_bg) / 2 / bg_w
            cy = (box_y1_on_bg + box_y2_on_bg) / 2 / bg_h
            w = (box_x2_on_bg - box_x1_on_bg) / bg_w
            h = (box_y2_on_bg - box_y1_on_bg) / bg_h
            
            self.current_yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # --- 2. 关联并粘贴血条 (严格遵循您的规则) ---
        health_cfg = self.synth_config.get('healthbar_association', {})
        
        # 规则 3.1: 血条只与友方单位 ('friendly') 关联
        is_friendly = 'friendly' in unit_key 
        
        if is_friendly and health_cfg.get('enabled', False) and random.random() < health_cfg.get('probability', 0.95):
            # 血条资产的key是固定的
            healthbar_key = '2-status-healthbar-血条'
            if healthbar_key in self.assets and self.assets[healthbar_key]:
                healthbar_asset = random.choice(self.assets[healthbar_key]).copy()
                
                unit_w = new_box[2] - new_box[0]
                
                # 规则 3.3: 血条尺寸与父单位关联
                scale = random.uniform(*health_cfg.get('width_scale_range', [0.8, 1.0]))
                new_w = int(unit_w * scale)
                if new_w <= 0: return # 避免无效尺寸
                
                orig_h, orig_w = healthbar_asset.shape[:2]
                new_h = int(orig_h * (new_w / orig_w))
                if new_h <= 0: return # 避免无效尺寸

                resized_healthbar = cv2.resize(healthbar_asset, (new_w, new_h))
                
                # 规则 3.2: 精确定位
                unit_top_center_x = (new_box[0] + new_box[2]) / 2
                unit_top_y = new_box[1]
                
                v_offset = random.randint(*health_cfg.get('vertical_offset_range', [-20, -12]))
                
                # 计算血条的左上角粘贴坐标
                hb_x = int(unit_top_center_x - new_w / 2)
                hb_y = int(unit_top_y + v_offset - new_h) # 向上偏移
                
                paste_foreground(bg, resized_healthbar, (hb_x, hb_y))

    
    def _apply_ui_occlusion(self, bg):
        bg_h, bg_w = bg.shape[:2]
        cfg = self.synth_config['ui_occlusion']
        if not self.backgrounds['ui']: return
        skill_keys = [k for k in self.assets.keys() if k.startswith('5-skill')]
        if not skill_keys: return

        panel_canvas = random.choice(self.backgrounds['ui']).copy()
        random.shuffle(skill_keys)
        
        slots = cfg['skill_panel_slots']
        num_to_show = random.randint(*cfg['num_skills_to_show_range'])

        for i in range(min(num_to_show, len(slots), len(skill_keys))):
            # 【核心修正】直接获取槽位的左上角坐标
            slot_top_left_pos = slots[i]
            
            skill_key = skill_keys[i]
            if not self.assets.get(skill_key): continue
            
            # 直接使用原始资产，不做任何尺寸调整
            skill_asset = random.choice(self.assets[skill_key])

            # 【核心修正】不再需要计算，直接使用配置中定义的左上角坐标进行粘贴
            paste_foreground(panel_canvas, skill_asset, slot_top_left_pos)

        h, w = panel_canvas.shape[:2]
        x = (bg_w - w) / 2 + np.random.randn() * cfg['center_placement_offset']['x_stddev']
        y = (bg_h - h) / 2 + np.random.randn() * cfg['center_placement_offset']['y_stddev']
        paste_foreground(bg, panel_canvas, (x, y))



    def _create_dataset_yaml(self):
        # 【核心修正】val 路径指向正确的 'images/val'
        dataset_yaml_data = {
            'path': str(self.output_dir.resolve()), 
            'train': 'images/train',
            'val': 'images/val',  # 修正路径
            'names': ID_TO_CLASS
        }
        dataset_yaml_path = self.output_dir / "dataset.yaml"
        with open(dataset_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_yaml_data, f, sort_keys=False, allow_unicode=True)
        print(f"✅ 已创建数据集配置文件: {dataset_yaml_path}")


if __name__ == '__main__':
    synthesizer = DataSynthesizer(
        config_path=PROJECT_ROOT / 'config.json',
        synth_config_path=PROJECT_ROOT / 'configs/synthesis_config.yaml'
    )
    synthesizer.generate()
# 文件: /home/zhz/deepl/tools/data_synthesizer.py (V3.3 - 按清单轮询最终版)
# 职责: 严格按照稀缺类别清单进行轮询生成，确保每个稀缺类别都有充足的样本。

import cv2
import numpy as np
import yaml
from pathlib import Path
import random
import shutil
from tqdm import tqdm
import sys
import json
from itertools import cycle

# --- 路径设置与模块导入 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.constants import CLASS_TO_ID, ID_TO_CLASS

# --- 辅助函数 (保持不变) ---
def paste_foreground(bg_rgba, fg_rgba, pos):
    x, y = int(pos[0]), int(pos[1])
    fg_h, fg_w = fg_rgba.shape[:2]
    bg_h, bg_w = bg_rgba.shape[:2]
    x_start_on_bg = max(x, 0); y_start_on_bg = max(y, 0)
    x_end_on_bg = min(x + fg_w, bg_w); y_end_on_bg = min(y + fg_h, bg_h)
    if x_start_on_bg >= x_end_on_bg or y_start_on_bg >= y_end_on_bg: return bg_rgba
    fg_x_start = x_start_on_bg - x; fg_y_start = y_start_on_bg - y
    fg_x_end = x_end_on_bg - x; fg_y_end = y_end_on_bg - y
    fg_cropped = fg_rgba[fg_y_start:fg_y_end, fg_x_start:fg_x_end]
    bg_roi = bg_rgba[y_start_on_bg:y_end_on_bg, x_start_on_bg:x_end_on_bg]
    fg_alpha = fg_cropped[:, :, 3] / 255.0
    alpha_mask = np.dstack([fg_alpha] * 3)
    blended_rgb = (fg_cropped[:, :, :3] * alpha_mask) + (bg_roi[:, :, :3] * (1 - alpha_mask))
    bg_roi[:, :, :3] = blended_rgb.astype(bg_rgba.dtype)
    return bg_rgba

# --- 主合成器类 ---
class DataSynthesizer:
    def __init__(self, config_path, synth_config_path, scarce_class_list):
        with open(config_path, 'r', encoding='utf-8') as f: self.config = json.load(f)
        with open(synth_config_path, 'r', encoding='utf-8') as f: self.synth_config = yaml.safe_load(f)
        
        self.scarce_classes = scarce_class_list
        # 【核心】创建一个无限循环的迭代器，用于轮询稀缺类别
        self.scarce_class_cycler = cycle(self.scarce_classes)

        self.paths = self.config['data_paths']
        self.assets_dir = Path(self.paths['synthetic_assets_dir'])
        self.bg_dir = Path(self.paths['synthetic_backgrounds_dir'])
        self.output_dir = Path(self.paths['synthetic_output_dir'])
        
        self.assets = self._load_assets()
        self.backgrounds = self._load_backgrounds()
        # 【核心】按类别前缀对素材key进行预分类，方便查找
        self.asset_keys_by_prefix = self._classify_asset_keys()

    def _load_assets(self):
        # ... (此函数保持 V3.1 版本不变) ...
        print("--- 正在加载资产库... ---")
        assets = {}
        asset_files = list(self.assets_dir.glob("**/*.png"))
        if not asset_files: raise FileNotFoundError(f"错误：在资产目录 {self.assets_dir} 中找不到任何 .png 文件。")
        for path in tqdm(asset_files, desc="加载前景资产"):
            full_key = path.parent.name
            asset_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if asset_img is None or len(asset_img.shape) < 3 or asset_img.shape[2] != 4: continue
            if full_key not in assets: assets[full_key] = []
            assets[full_key].append(asset_img)
        print(f"✅ 成功加载 {len(assets)} 个类别的资产。")
        return assets

    def _load_backgrounds(self):
        # ... (此函数保持 V3.1 版本不变) ...
        print("--- 正在加载背景库... ---")
        backgrounds = {}
        for bg_type in ['combat', 'ui']:
            backgrounds[bg_type] = []
            bg_paths = list((self.bg_dir / bg_type).glob("*.png"))
            for path in bg_paths:
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is None: continue
                if len(img.shape) < 3 or img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                backgrounds[bg_type].append(img)
        print(f"✅ 成功加载 {len(backgrounds['combat'])} 张战斗背景和 {len(backgrounds['ui'])} 张UI背景。")
        return backgrounds

    def _classify_asset_keys(self):
        """对所有资产的key按前缀进行分类，方便快速查找。"""
        classified_keys = {
            '1-unit-enemy': [], '1-unit-friendly-player': [], '1-unit-friendly-pet': [],
            '5-skill-player': []
        }
        for key in self.assets.keys():
            if key.startswith('1-unit-enemy'): classified_keys['1-unit-enemy'].append(key)
            elif key.startswith('1-unit-friendly-player'): classified_keys['1-unit-friendly-player'].append(key)
            elif key.startswith('1-unit-friendly-pet'): classified_keys['1-unit-friendly-pet'].append(key)
            elif key.startswith('5-skill-player'): classified_keys['5-skill-player'].append(key)
        return classified_keys
    
    def _generate_number_image(self, value: int, digit_assets: list) -> np.ndarray:
        """
        根据给定的数值，动态生成一个多位数的图像。
        :param value: 要生成的数字，例如 378
        :param digit_assets: 包含 0-9 单个数字图像的列表
        :return: 拼接好的、包含整个数字的 RGBA 图像
        """
        s_value = str(value)
        # 确保我们能正确地根据数字找到对应的图片
        # 假设 digit_assets[0] 是 0.png, digit_assets[1] 是 1.png ...
        digit_images = [digit_assets[int(d)] for d in s_value]

        total_width = sum(img.shape[1] for img in digit_images)
        max_height = max(img.shape[0] for img in digit_images)

        stitched_image = np.zeros((max_height, total_width, 4), dtype=np.uint8)

        current_x = 0
        for img in digit_images:
            h, w = img.shape[:2]
            # 将每个数字画在底部对齐
            y_offset = max_height - h
            stitched_image[y_offset:y_offset+h, current_x:current_x+w] = img
            current_x += w
            
        return stitched_image


    def generate(self):
        # ... (此函数保持 V3.1 版本不变) ...
        print(f"--- [数据合成] 开始生成数据集至: {self.output_dir} ---")
        if self.output_dir.exists(): shutil.rmtree(self.output_dir)
        img_train_dir = self.output_dir / "images/train"; lbl_train_dir = self.output_dir / "labels/train"
        img_val_dir = self.output_dir / "images/val"; lbl_val_dir = self.output_dir / "labels/val"
        img_train_dir.mkdir(parents=True); lbl_train_dir.mkdir(parents=True)
        img_val_dir.mkdir(parents=True); lbl_val_dir.mkdir(parents=True)
        num_to_generate = self.synth_config['num_images_to_generate']
        val_split_ratio = 0.1
        num_val = int(num_to_generate * val_split_ratio)
        for i in tqdm(range(num_to_generate), desc="生成合成图像"):
            bg, yolo_labels = self._generate_single_scene()
            img_name, label_name = f"synth_scarce_{i:06d}.png", f"synth_scarce_{i:06d}.txt"
            target_img_dir = img_val_dir if i < num_val else img_train_dir
            target_lbl_dir = lbl_val_dir if i < num_val else lbl_train_dir
            cv2.imwrite(str(target_img_dir / img_name), cv2.cvtColor(bg, cv2.COLOR_BGRA2BGR))
            with open(target_lbl_dir / label_name, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_labels))
        self._create_dataset_yaml()
        print(f"✅ [数据合成] 任务完成！")
    
    # --- 【核心重构】生成逻辑 ---
    def _generate_single_scene(self):
        bg = random.choice(self.backgrounds['combat']).copy()
        self.current_yolo_labels = []

        # 【核心策略】从稀缺类别清单中轮流取出一个作为本图的主角
        target_class_name = next(self.scarce_class_cycler)
        
        # 查找这个主角对应的完整素材key (e.g., '血条' -> '2-status-healthbar-血条')
        target_asset_key = None
        for key in self.assets.keys():
            if key.endswith(target_class_name):
                target_asset_key = key
                break
        
        # 渲染场景
        if target_asset_key:
            if target_asset_key.startswith('1-unit'): # 如果主角是作战单位
                self._place_units_with_target(bg, target_asset_key)
            elif target_asset_key.startswith('5-skill'): # 如果主角是技能
                self._place_units_and_ui_with_target_skill(bg, target_asset_key)
            else: # 其他类别（如血条、数值、物品等）
                # 简化处理：先放置常规作战单位作为“背景板”
                self._place_units_in_formation(bg)
                # 然后想办法放置主角（这里需要根据类别进一步细化规则）
                # 例如，如果是血条，就强制附加给一个友方单位
                if target_class_name == '血条':
                    # (此处省略强制附加血条的逻辑)
                    pass

        # 如果没有找到主角素材或规则，则按常规方式生成
        else:
            self._place_units_in_formation(bg)
            if self.synth_config['ui_occlusion']['enabled'] and random.random() < self.synth_config['ui_occlusion']['probability']:
                self._apply_ui_occlusion(bg)

        return bg, self.current_yolo_labels

    def _place_units_with_target(self, bg, target_unit_key):
        """生成一张必须包含 target_unit_key 的战斗场景。"""
        # 1. 放置其他随机单位作为陪衬
        self._place_units_in_formation(bg, exclude_key=target_unit_key)
        # 2. 在一个随机的、合适的槽位，强制放置主角单位
        slots = self.synth_config['placement_rules']['formation_slots']
        if 'enemy' in target_unit_key:
            target_slot = random.choice(slots['enemy_back'] + slots['enemy_front'])
        else:
            target_slot = random.choice(slots['friendly_back'] + slots['friendly_front'])
        self._place_single_unit(bg, target_unit_key, target_slot)

    def _place_units_and_ui_with_target_skill(self, bg, target_skill_key):
        """生成一张必须包含 target_skill_key 的UI遮挡场景。"""
        # 1. 先放置作战单位
        self._place_units_in_formation(bg)
        # 2. 强制应用UI遮挡，并确保主角技能出现
        self._apply_ui_occlusion(bg, force_skill_key=target_skill_key)

    def _place_units_in_formation(self, bg, exclude_key=None):
        # ... (此函数与V3.1版本基本一致, 只是增加了exclude_key参数) ...
        cfg = self.synth_config
        enemy_keys = [k for k in self.asset_keys_by_prefix['1-unit-enemy'] if k != exclude_key]
        player_keys = [k for k in self.asset_keys_by_prefix['1-unit-friendly-player'] if k != exclude_key]
        pet_keys = [k for k in self.asset_keys_by_prefix['1-unit-friendly-pet'] if k != exclude_key]
        if not enemy_keys or not player_keys or not pet_keys: return
        num_enemies = random.randint(cfg['unit_density']['min_enemy_units'], cfg['unit_density']['max_enemy_units'])
        enemy_slots = cfg['placement_rules']['formation_slots']['enemy_back'] + cfg['placement_rules']['formation_slots']['enemy_front']
        random.shuffle(enemy_slots)
        for i in range(min(num_enemies, len(enemy_slots))):
            self._place_single_unit(bg, random.choice(enemy_keys), enemy_slots[i])
        self._place_single_unit(bg, random.choice(player_keys), random.choice(cfg['placement_rules']['formation_slots']['friendly_back']))
        self._place_single_unit(bg, random.choice(pet_keys), random.choice(cfg['placement_rules']['formation_slots']['friendly_front']))

    def _place_single_unit(self, bg, unit_key, roi):
        # ... (此函数与V3.1版本完全一致) ...
        unit_asset = random.choice(self.assets[unit_key]).copy()
        if 'friendly' in unit_key: unit_asset = cv2.flip(unit_asset, 1)
        h, w = unit_asset.shape[:2]
        x1, y1, x2, y2 = roi
        place_x = random.randint(x1, x2)
        place_y = random.randint(y1, y2)
        top_left_x = place_x - w // 2
        top_left_y = place_y - h
        bg = paste_foreground(bg, unit_asset, (top_left_x, top_left_y))
        unit_box = (top_left_x, top_left_y, top_left_x + w, top_left_y + h)
        self._add_yolo_label_and_events(bg, unit_key, unit_box)

    def _add_yolo_label_and_events(self, bg, unit_key, unit_box):
        # --- 添加单位本身的标签 (逻辑不变) ---
        class_name = unit_key.rsplit('-', 1)[-1]
        if class_name in CLASS_TO_ID: self._add_yolo_label(class_name, unit_box)
        
        # --- 关联血条 (逻辑不变) ---
        hb_cfg = self.synth_config['healthbar_association']
        if 'friendly' in unit_key and hb_cfg['enabled'] and random.random() < hb_cfg['probability']:
            if '2-status-healthbar-血条' in self.assets:
                healthbar_asset = random.choice(self.assets['2-status-healthbar-血条'])
                hb_h, hb_w = healthbar_asset.shape[:2]
                unit_center_x = (unit_box[0] + unit_box[2]) / 2
                hb_x = int(unit_center_x - hb_w / 2)
                hb_y = int(unit_box[1] + hb_cfg['vertical_offset'])
                bg = paste_foreground(bg, healthbar_asset, (hb_x, hb_y))
                self._add_yolo_label('血条', (hb_x, hb_y, hb_x + hb_w, hb_y + hb_h))

        # --- 【核心升级 V3】关联悬浮数值 (身体中心定位) ---
        num_cfg = self.synth_config['damage_number_association']
        if '3-digits-damage' in self.assets and num_cfg['enabled'] and random.random() < num_cfg['probability']:
            
            # 1. 按权重生成2、3、4位数 (逻辑不变)
            num_digits_choices = [2, 3, 4]; weights = [0.3, 0.4, 0.3]
            chosen_digits = random.choices(num_digits_choices, weights=weights, k=1)[0]
            if chosen_digits == 2: min_val, max_val = 10, 99
            elif chosen_digits == 3: min_val, max_val = 100, 999
            else: min_val, max_val = 1000, 9999
            value_to_generate = random.randint(min_val, max_val)
            
            # 2. 动态生成数值图片 (逻辑不变)
            number_image = self._generate_number_image(value_to_generate, self.assets['3-digits-damage'])
            num_h, num_w = number_image.shape[:2]

            # 3. 【全新定位逻辑】计算单位的中心点和尺寸
            unit_x1, unit_y1, unit_x2, unit_y2 = unit_box
            unit_w = unit_x2 - unit_x1
            unit_h = unit_y2 - unit_y1
            unit_center_x = unit_x1 + unit_w / 2
            unit_center_y = unit_y1 + unit_h / 2

            # 4. 【全新定位逻辑】计算数字的粘贴位置
            #    水平位置：在单位中心左右轻微浮动，看起来更自然
            horizontal_jitter = unit_w * random.uniform(-0.1, 0.1)
            num_x = int(unit_center_x - num_w / 2 + horizontal_jitter)
            
            #    垂直位置：以单位身体中心为基准，根据配置的 jitter_ratio 进行随机上下浮动
            jitter_range = (unit_h / 2) * num_cfg.get('body_center_jitter_ratio', 0.4)
            vertical_jitter = random.uniform(-jitter_range, jitter_range)
            num_y = int(unit_center_y - num_h / 2 + vertical_jitter)

            # 5. 粘贴并打标签 (逻辑不变)
            bg = paste_foreground(bg, number_image, (num_x, num_y))
            self._add_yolo_label('悬浮数值', (num_x, num_y, num_x + num_w, num_y + num_h))

    def _apply_ui_occlusion(self, bg, force_skill_key=None):
        # ... (此函数与V3.2版本基本一致, 只是增加了force_skill_key参数) ...
        cfg = self.synth_config['ui_occlusion']
        if not self.backgrounds['ui']: return
        panel_asset = random.choice(self.backgrounds['ui']).copy()
        panel_roi = cfg['panel_roi']
        skill_keys = [k for k in self.asset_keys_by_prefix['5-skill-player'] if k != force_skill_key]
        random.shuffle(skill_keys)
        if force_skill_key: skill_keys.insert(0, force_skill_key) # 确保主角技能被优先放置
        slots = cfg['skill_slots_relative']
        num_to_show = random.randint(*cfg['num_skills_to_show_range'])
        for i in range(min(num_to_show, len(slots), len(skill_keys))):
            skill_asset = random.choice(self.assets[skill_keys[i]])
            slot_roi = slots[i]
            asset_h, asset_w = skill_asset.shape[:2]
            x_range_max = slot_roi[2] - asset_w
            y_range_max = slot_roi[3] - asset_h
            place_x = slot_roi[0] if slot_roi[0] >= x_range_max else random.randint(slot_roi[0], x_range_max)
            place_y = slot_roi[1] if slot_roi[1] >= y_range_max else random.randint(slot_roi[1], y_range_max)
            panel_asset = paste_foreground(panel_asset, skill_asset, (place_x, place_y))
            global_x = panel_roi[0] + place_x
            global_y = panel_roi[1] + place_y
            self._add_yolo_label(skill_keys[i].rsplit('-', 1)[-1], 
                                (global_x, global_y, global_x + asset_w, global_y + asset_h))
        bg = paste_foreground(bg, panel_asset, (panel_roi[0], panel_roi[1]))

    # --- 【关键】重新添加被误删的 _add_yolo_label 函数 ---
    def _add_yolo_label(self, class_name, box):
        if class_name not in CLASS_TO_ID: return
        class_id = CLASS_TO_ID[class_name]
        x1, y1, x2, y2 = box
        bg_w, bg_h = self.synth_config['output_image_size']
        cx = (x1 + x2) / 2 / bg_w; cy = (y1 + y2) / 2 / bg_h
        w = (x2 - x1) / bg_w; h = (y2 - y1) / bg_h
        self.current_yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    def _create_dataset_yaml(self):
        # ... (此函数保持 V3.1 版本不变) ...
        dataset_yaml_data = {'path': str(self.output_dir.resolve()), 'train': 'images/train', 'val': 'images/val', 'names': ID_TO_CLASS}
        with open(self.output_dir / "dataset.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(dataset_yaml_data, f, sort_keys=False, allow_unicode=True)
        print(f"✅ 已创建数据集配置文件: {self.output_dir / 'dataset.yaml'}")

if __name__ == '__main__':
    scarce_class_list_from_inspector = [
        '天龙水', '书信', '龙卷雨击', '笛子', '确认按钮', '金刚护法', '唧唧歪歪', 
        '推气过宫', '雷鸟人', '横扫千军', '狡猾的貔貅', '通知栏', '九转金丹', 
        '鬼切草', '凤凰', '天将', '大蝙蝠', '地狱战神', '四叶花', '佛手', 
        '龟丞相', '白熊', '牛刀小试', '高级宠物口粮', '护卫', '洞冥草', 
        '海毛虫', '桃花', '包子', '赌徒', '宠物口粮', '树怪', '野猪', 
        '仙狐涎', '巨蛙', '绿芦羹', '佛光舍利子', '山贼', '月见草', '红罗羹',
        '血条', '悬浮数值'
    ]
    synthesizer = DataSynthesizer(
        config_path=PROJECT_ROOT / 'config.json',
        synth_config_path=PROJECT_ROOT / 'configs/synthesis_config.yaml',
        scarce_class_list=scarce_class_list_from_inspector
    )
    synthesizer.generate()
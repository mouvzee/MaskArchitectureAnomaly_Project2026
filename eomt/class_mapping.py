# class_mapping.py
import numpy as np
import torch

# Cityscapes 19 train_id names (index = train_id)
CITYSCAPES_LABELS = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

# COCO reindexed id (model output) → Cityscapes train_id
# Using verified values from CLASS_MAPPING in coco_panoptic.py
COCO_TO_CITYSCAPES = {
    # --- THINGS ---
    0:  11,  # person        → person (11)
    1:  18,  # bicycle       → bicycle (18)
    2:  13,  # car           → car (13)
    3:  17,  # motorcycle    → motorcycle (17)
    5:  15,  # bus           → bus (15)
    6:  16,  # train         → train (16)
    7:  14,  # truck         → truck (14)
    9:   6,  # traffic light → traffic light (6)
    11:  7,  # stop sign     → traffic sign (7)

    # --- STUFF ---
    87:  0,  # pavement      → road (0)
    88:  1,  # sidewalk      → sidewalk (1)
    93:  2,  # building      → building (2)
    95:  3,  # wall          → wall (3)
    97:  4,  # fence         → fence (4)
    96:  5,  # pole          → pole (5)
    100: 8,  # tree          → vegetation (8)
    91:  9,  # dirt          → terrain (9)
    110: 10, # sky           → sky (10)
}

# Lookup table: built ONCE at import time
# Size 201 to safely cover all reindexed ids (max is 132, but 201 for safety)
LOOKUP_TABLE = torch.full((201,), 255, dtype=torch.long)
for coco_id, city_id in COCO_TO_CITYSCAPES.items():
    LOOKUP_TABLE[coco_id] = city_id

def remap_coco_to_cityscapes(pred):
    """
    Args:
        pred: torch.Tensor of shape (H,W) or (B,H,W) with COCO reindexed ids
    Returns:
        torch.Tensor same shape with Cityscapes train_ids (255 = ignore)
    """
    global LOOKUP_TABLE
    if LOOKUP_TABLE.device != pred.device:
        LOOKUP_TABLE = LOOKUP_TABLE.to(pred.device)
    return LOOKUP_TABLE[pred.long()]

# Common class info
COMMON_TRAIN_IDS = sorted(set(COCO_TO_CITYSCAPES.values()))
COMMON_CLASS_NAMES = [CITYSCAPES_LABELS[i] for i in COMMON_TRAIN_IDS]
NUM_COMMON_CLASSES = len(COMMON_TRAIN_IDS)  # 14
# class_mapping.py
import numpy as np
import torch

# Cityscapes 19 train_id names (index = train_id)
CITYSCAPES_LABELS = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

# COCO category_id → Cityscapes train_id
COCO_TO_CITYSCAPES = {
    # --- THINGS (Indices 0 - 79) ---
    0:  11,  # person        -> person (11)
    1:  18,  # bicycle       -> bicycle (18)
    2:  13,  # car           -> car (13)
    3:  17,  # motorcycle    -> motorcycle (17)
    5:  15,  # bus           -> bus (15)
    6:  16,  # train         -> train (16)
    7:  14,  # truck         -> truck (14)
    9:   6,  # traffic light -> traffic light (6)
    11:  7,  # stop sign     -> traffic sign (7)

    # --- STUFF (Indices 80 - 132) ---
    
    # Road (0) & Sidewalk (1)
    125: 0,  # pavement      -> road (0)
    
    # Building (2)
    84:  2,  # building-other -> building (2)
    113: 2,  # house         -> building (2)
    
    # Fence (4)
    98:  4,  # fence         -> fence (4)
    
    # Vegetation (8)
    82:  8,  # branch        -> vegetation (8)
    85:  8,  # bush          -> vegetation (8)
    104: 8,  # flower        -> vegetation (8)
    109: 8,  # grass         -> vegetation (8)
    114: 8,  # leaves        -> vegetation (8)
    127: 8,  # plant-other   -> vegetation (8)
    
    # Terrain (9)
    96:  9,  # dirt          -> terrain (9)
    110: 9,  # gravel        -> terrain (9)
    111: 9,  # ground-other  -> terrain (9)
    112: 9,  # hill          -> terrain (9)
    120: 9,  # mountain      -> terrain (9)
    121: 9,  # mud           -> terrain (9)
    
    # Sky (10)
    91:  10, # clouds        -> sky (10)
}


# Lookup table: built ONCE at import time
LOOKUP_TABLE = torch.full((256,), 255, dtype=torch.long)
for coco_id, city_id in COCO_TO_CITYSCAPES.items():
    LOOKUP_TABLE[coco_id] = city_id

def remap_coco_to_cityscapes(pred):
    """
    Args:
        pred: torch.Tensor of shape (H,W) or (B,H,W) with COCO category_ids
    Returns:
        torch.Tensor same shape with Cityscapes train_ids (255 = ignore)
    """
    global LOOKUP_TABLE
    
    if LOOKUP_TABLE.device != pred.device:
        LOOKUP_TABLE = LOOKUP_TABLE.to(pred.device)

    return LOOKUP_TABLE[pred.long()]

# Common class info (14 classes that have a mapping)
COMMON_TRAIN_IDS = sorted(set(COCO_TO_CITYSCAPES.values()))
COMMON_CLASS_NAMES = [CITYSCAPES_LABELS[i] for i in COMMON_TRAIN_IDS]
NUM_COMMON_CLASSES = len(COMMON_TRAIN_IDS)  # 14
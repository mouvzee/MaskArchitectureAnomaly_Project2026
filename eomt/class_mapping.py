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
    1:   11,  # person        → person
    2:   18,  # bicycle       → bicycle
    3:   13,  # car           → car
    4:   17,  # motorcycle    → motorcycle
    6:   15,  # bus           → bus
    7:   16,  # train         → train
    8:   14,  # truck         → truck
    10:   6,  # traffic light → traffic light
    13:   7,  # stop sign     → traffic sign
    118:  0,  # pavement      → road
    119:  1,  # sidewalk      → sidewalk
    134:  2,  # building      → building
    158:  8,  # tree          → vegetation
    167: 10,  # sky           → sky
    128:  9,  # dirt          → terrain
    145:  4,  # fence         → fence
    135:  3,  # wall          → wall
    144:  5,  # pole          → pole
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
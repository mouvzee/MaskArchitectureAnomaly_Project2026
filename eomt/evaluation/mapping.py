# Cityscapes classes (0-18):
#  0 road       | 1 sidewalk   | 2 building  | 3 wall    | 4 fence
#  5 pole       | 6 traf.light | 7 traf.sign | 8 vegetat.| 9 terrain
# 10 sky        | 11 person    | 12 rider    | 13 car    | 14 truck
# 15 bus        | 16 train     | 17 motorcycle | 18 bicycle

IGNORE_INDEX = 255

# Maps contiguous COCO panoptic IDs (0-132: 80 things + 53 stuff)
# to Cityscapes class IDs (0-18). Unmapped IDs → IGNORE_INDEX.
COCO_TO_CITYSCAPES: dict[int, int] = {
    # ── COCO Things ───────────────────────────────────────
    0:  11,  # person        → person
    1:  18,  # bicycle       → bicycle
    2:  13,  # car           → car
    3:  17,  # motorcycle    → motorcycle
    5:  15,  # bus           → bus
    6:  16,  # train         → train
    7:  14,  # truck         → truck
    9:   6,  # traffic light → traf.light
    11:  7,  # stop sign     → traf.sign

    # ── COCO Stuff ────────────────────────────────────────
    100:  0,  # road
    123:  1,  # pavement/sidewalk
    91:   2,  # building
    101:  2,  # house
    129:  2,  # skyscraper
    109:  3,  # wall-brick
    110:  3,  # wall-stone
    111:  3,  # wall-tile
    112:  3,  # wall-wood
    131:  3,  # wall-other
    117:  4,  # fence
    116:  8,  # vegetation/tree
    125:  9,  # terrain/grass
    126:  9,  # dirt
    102:  9,  # gravel
    119: 10,  # sky
}
### mIoU Evaluation on Cityscapes

Evaluates semantic segmentation performance. Must be run from the `eomt/` 
directory. Two YAML configs are provided in `eomt/configs/`.

**Cityscapes model:**
```bash
cd eomt

python evaluation/run_eval.py \
    --config  configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
    --data    /path/to/cityscapes \
    --ckpt    /path/to/eomt_cityscapes.pth \
    --city-config  configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
    --city-ckpt    /path/to/eomt_cityscapes.pth
```

**COCO model** (adds automatic COCO → Cityscapes class remapping):
```bash
cd eomt

python evaluation/run_eval.py \
    --config  configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml \
    --data    /path/to/coco \
    --ckpt    /path/to/eomt_coco.pth \
    --city-config  configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
    --city-ckpt    /path/to/eomt_cityscapes.pth \
    --coco-model
```

> ⚠️ Do NOT run from the repo root — Python will raise 
> `ModuleNotFoundError: No module named 'evaluation'`.
> Always `cd eomt` first.
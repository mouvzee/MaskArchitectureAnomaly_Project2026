# utils_eomt.py
import yaml
import importlib
import warnings
 
import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torchmetrics import JaccardIndex
 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from lightning import seed_everything
 
from .mapping import COCO_TO_CITYSCAPES, IGNORE_INDEX


def setup_environment(seed: int = 0) -> None:
    seed_everything(seed, verbose=False)
    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module`.*",
    )


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# LOAD MODEL, DATA, AND CHECKPOINT
def load_model_and_data(config, data_path,ckpt_path, device):
    # LOAD DATASET
    data_module_name, class_name = config["data"]["class_path"].rsplit(".", 1)
    data_module = getattr(importlib.import_module(data_module_name), class_name)
    data_module_kwargs = config["data"].get("init_args", {})

    data = data_module(
        path=data_path,
        batch_size=1,
        num_workers=0,
        check_empty_targets=False,
        **data_module_kwargs
    ).setup()

    # Load encoder
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    encoder = encoder_cls(img_size=data.img_size, **encoder_cfg.get("init_args", {}))

    # Load network
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=data.num_classes,
        encoder=encoder,
        **network_kwargs,
    )

    # Load Lightning module
    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    if "stuff_classes" in config["data"].get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

    model = (
        lit_cls(
            img_size=data.img_size,
            num_classes=data.num_classes,
            network=network,
            **model_kwargs,
            )
        .eval()
        .to(device)
        )
    
    # LOAD WEIGHTS
    ckpt = torch.load(ckpt_path, map_location=f"cuda:{device}", weights_only=False)
    state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)

    print(f"Model and data loaded successfully from {ckpt_path}")

    return model, data

# MAPPING COCO → CITYSCAPES
def remap_coco_to_cityscapes(pred_array: np.ndarray) -> np.ndarray:
    """
    Remap a 2D array of COCO predictions (0-132)
    to Cityscapes classes (0-18). Unmapped IDs → IGNORE_INDEX.
    """
    out = np.full_like(pred_array, fill_value=IGNORE_INDEX, dtype=np.int64)
    for coco_id, city_id in COCO_TO_CITYSCAPES.items():
        out[pred_array == coco_id] = city_id
    return out

# EVALUATE THE MODELS
def evaluate(model, dataloader, data, DEVICE, is_coco_model: bool, city_model):
    """Compute mIoU over 19 Cityscapes classes.
 
    Args:
        model: the model to evaluate.
        dataloader: Cityscapes val dataloader (always, regardless of model type).
        data: data module (for img_size).
        device: CUDA device index.
        is_coco_model: if True, remaps predictions from COCO space to Cityscapes.
        city_model: Cityscapes model used to convert targets.
 
    Returns:
        mIoU as a float in [0, 1].
    """

    metric = JaccardIndex(
        task="multiclass",
        num_classes=19,
        ignore_index=255,
        average="macro",
    ).to(DEVICE)

    model.eval()

    if is_coco_model:
      mapped_count = sum(1 for v in COCO_TO_CITYSCAPES.values() if v != IGNORE_INDEX)
      print(f"COCO classes mapped on Cityscapes: {mapped_count} / 133")
      
    for img, target in tqdm(dataloader):
        img    = img[0]
        target = target[0]

        with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
            imgs      = [img.to(DEVICE)]
            img_sizes = [img.shape[-2:]]
            crops, origins = model.window_imgs_semantic(imgs)
            mask_logits_per_layer, class_logits_per_layer = model(crops)
            mask_logits = F.interpolate(
                mask_logits_per_layer[-1], data.img_size, mode="bilinear"
            )
            crop_logits = model.to_per_pixel_logits_semantic(
                mask_logits, class_logits_per_layer[-1]
            )
            logits_list = model.revert_window_logits_semantic(
                crop_logits, origins, img_sizes
            )

        preds = logits_list[0].argmax(dim=0).cpu()

        if is_coco_model:
            preds = torch.from_numpy(remap_coco_to_cityscapes(preds.numpy())).long()
        

        gt = city_model.to_per_pixel_targets_semantic([target], 255)[0]

        # Maschera pixel ignorati
        preds_flat = preds.reshape(-1)
        gt_flat    = gt.reshape(-1)
        mask       = (gt_flat != 255) & (preds_flat != 255)
        preds_flat = preds_flat[mask]
        gt_flat    = gt_flat[mask]

        metric.update(preds_flat[None].to(DEVICE), gt_flat[None].to(DEVICE))

    miou = metric.compute()
    print(f"mIoU: {miou * 100:.1f}%")
    return miou.item()
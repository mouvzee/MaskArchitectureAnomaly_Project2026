"""
Evaluate an EoMT model (Cityscapes or COCO) on the Cityscapes val set.

Usage:
    python run_eval.py \
        --config  configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
        --data    /path/to/cityscapes \
        --ckpt    checkpoints/eomt_cityscapes.bin \
        --city-config  configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
        --city-ckpt    checkpoints/eomt_cityscapes.bin

    # For a COCO model (adds COCO→Cityscapes remapping):
    python run_eval.py ... --coco-model \
        --config  configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml \
        --data    /path/to/coco_data \
        --ckpt    checkpoints/eomt_coco.bin
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from evaluation import load_config, load_model_and_data, setup_environment, evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EoMT evaluation on Cityscapes")
    p.add_argument("--config",      required=True, help="YAML config of the model to evaluate")
    p.add_argument("--data",        required=True, help="Path to the model's dataset")
    p.add_argument("--ckpt",        required=True, help="Checkpoint of the model to evaluate")
    p.add_argument("--city-config", required=True, help="YAML config of the Cityscapes reference model")
    p.add_argument("--city-ckpt",   required=True, help="Checkpoint of the Cityscapes reference model")
    p.add_argument("--city-data",   default=None, help="Path to Cityscapes data (if different from --data)")
    p.add_argument("--device",      type=int, default=0, help="CUDA device index (default: 0)")
    p.add_argument("--seed",        type=int, default=0, help="Random seed (default: 0)")
    p.add_argument("--coco-model",  action="store_true", help="Enable COCO→Cityscapes remapping")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_environment(seed=args.seed)

    config = load_config(args.config)
    model, data = load_model_and_data(config, args.data, args.ckpt, args.device)

    city_config = load_config(args.city_config)
    city_data_path = args.city_data if args.city_data else args.data
    city_model, city_data = load_model_and_data(city_config, city_data_path, args.city_ckpt, args.device)

    miou = evaluate(
        model=model,
        dataloader=city_data.val_dataloader(),
        data=data,
        DEVICE=args.device,
        is_coco_model=args.coco_model,
        city_model=city_model,
    )

    print(f"\nFinal mIoU: {miou * 100:.1f}%")


if __name__ == "__main__":
    main()
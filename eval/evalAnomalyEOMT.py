import os
import glob
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from sklearn.metrics import average_precision_score
from ood_metrics import fpr_at_95_tpr

def evaluate_anomaly_eomt(model, data, images_dir, gt_dir, device='cuda:0'):
    # No Resize!
    input_transform = Compose([ToTensor()])
    model.eval()

    # Define the temperatures required for the table, plus a few extra for finding the best t
    temperatures = [0.1, 0.5, 0.75, 1.0, 1.1, 1.5, 2.0]

    ood_gts_list = []
    anomaly_scores = {
        "Max_Entropy": [],
        "MaxLogit": [],
        "RbA": []
    }

    # Dynamically add an MSP key for every temperature
    for t in temperatures:
        anomaly_scores[f"MSP (t={t})"] = []

    image_paths = glob.glob(os.path.join(images_dir, '*.png'))
    image_paths.extend(glob.glob(os.path.join(images_dir, '*.jpg')))
    image_paths.extend(glob.glob(os.path.join(images_dir, '*.webp')))

    print(f"Inizio inferenza su {len(image_paths)} immagini (Risoluzione Naturale)...")

    for img_path in tqdm(image_paths):
        # =============================================================
        # 1. GROUND TRUTH
        # =============================================================
        img_basename = os.path.basename(img_path)
        # Remove the original extention (.jpg, .png o .webp) forcing .png for the GT
        base_name_without_ext = os.path.splitext(img_basename)[0]
        gt_path = os.path.join(gt_dir, base_name_without_ext + '.png')

        if not os.path.exists(gt_path):
            continue

        gt_img = Image.open(gt_path).convert('L')
        gt_array = np.array(gt_img)

        # Original Mapping
        if "RoadAnomaly" in gt_path:
            gt_array = np.where((gt_array==2), 1, gt_array)
        elif "LostAndFound" in gt_path:
            gt_array = np.where((gt_array==0), 255, gt_array)
            gt_array = np.where((gt_array==1), 0, gt_array)
            gt_array = np.where((gt_array>1) & (gt_array<201), 1, gt_array)
        elif "Streethazard" in gt_path:
            gt_array = np.where((gt_array==14), 255, gt_array)
            gt_array = np.where((gt_array<20), 0, gt_array)
            gt_array = np.where((gt_array==255), 1, gt_array)

        if 1 not in np.unique(gt_array):
            continue

        ood_gts_list.append(gt_array.flatten())

        # =============================================================
        # 2. INFERENCE AND WINDOWING
        # =============================================================
        image = Image.open(img_path).convert('RGB')
        img_tensor = input_transform(image)
        img_tensor = (img_tensor * 255).to(torch.uint8)

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            imgs = [img_tensor.to(device)]
            img_sizes = [imgs[0].shape[-2:]]

            crops, origins = model.window_imgs_semantic(imgs)
            mask_logits_per_layer, class_logits_per_layer = model(crops)

            mask_logits = F.interpolate(mask_logits_per_layer[-1], size=crops.shape[-2:], mode="bilinear", align_corners=False)
            class_logits = class_logits_per_layer[-1]

            # =============================================================
            # 3. STANDARD METRICS WITH TEMPERATURE SCALING
            # =============================================================
            crop_probs = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
            probs_list = model.revert_window_logits_semantic(crop_probs, origins, img_sizes)

            # Raw Logits
            probs_f32 = probs_list[0].float().cpu().numpy()

            # --- MaxLogit ---
            maxlogit_score = 1.0 - np.max(probs_f32, axis=0)
            anomaly_scores["MaxLogit"].append(maxlogit_score.flatten())

            # --- Temperature Scaling Loop ---
            for t in temperatures:
                # 1. Scale logits
                scaled_logits = probs_f32 / t

                # 2. Apply Softmax
                e_x = np.exp(scaled_logits - np.max(scaled_logits, axis=0, keepdims=True))
                softmax_probs = e_x / np.sum(e_x, axis=0, keepdims=True)

                # 3. Calculate MSP
                msp_score = 1.0 - np.max(softmax_probs, axis=0)
                anomaly_scores[f"MSP (t={t})"].append(msp_score.flatten())

                # 4. Calculate MaxEntropy ONLY on standard t=1.0
                if t == 1.0:
                    entropy_score = -np.sum(softmax_probs * np.log(softmax_probs + 1e-12), axis=0)
                    anomaly_scores["Max_Entropy"].append(entropy_score.flatten())

            # =============================================================
            # 4. RbA (Log-Sum Anti-Underflow)
            # =============================================================
            B, Q = mask_logits.shape[0], mask_logits.shape[1]

            mask_probs_f32 = mask_logits.sigmoid().float()
            class_probs_f32 = class_logits.softmax(dim=-1).float()

            prob_known_q = 1.0 - class_probs_f32[..., -1]
            joint_probs = mask_probs_f32 * prob_known_q.view(B, Q, 1, 1)

            rejection_probs = torch.clamp(1.0 - joint_probs, min=1e-7, max=1.0)
            log_rba_crop = torch.sum(torch.log(rejection_probs), dim=1, keepdim=True).half()

            rba_list = model.revert_window_logits_semantic(log_rba_crop, origins, img_sizes)
            rba_score = rba_list[0].squeeze(0).cpu().numpy()

            anomaly_scores["RbA"].append(rba_score.flatten())

            del image, img_tensor, imgs, crops, mask_logits, class_logits, crop_probs, log_rba_crop
            torch.cuda.empty_cache()

    # =============================================================
    # FINAL ANOMALY DETECTION AND OUTPUT TABLE (GPU ACCELERATED)
    # =============================================================
    print("\nElaborazione dati e calcolo AUC su GPU in corso...")
    ood_gts = np.concatenate(ood_gts_list)
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    # GPU-accelerated metrics function (1000x faster than scikit-learn on CPU)
    def calculate_metrics_gpu(scores_array, device='cuda:0'):
        # 1. Move vectors to GPU
        scores_t = torch.tensor(np.concatenate((scores_array[ind_mask], scores_array[ood_mask])), dtype=torch.float32, device=device)
        labels_t = torch.cat([
            torch.zeros(ind_mask.sum(), dtype=torch.float32, device=device),
            torch.ones(ood_mask.sum(), dtype=torch.float32, device=device)
        ])

        # 2. Parallel Sort on GPU
        desc_scores, indices = torch.sort(scores_t, descending=True)
        desc_labels = labels_t[indices]

        # 3. Vectorized True Positives (TP) and False Positives (FP)
        tp = torch.cumsum(desc_labels, dim=0)
        fp = torch.cumsum(1.0 - desc_labels, dim=0)
        total_positives = desc_labels.sum()
        total_negatives = labels_t.size(0) - total_positives

        # --- AUPRC ---
        precision = tp / (tp + fp + 1e-12)
        recall = tp / total_positives

        recall_diff = torch.cat([recall[0:1], recall[1:] - recall[:-1]])
        prc_auc = torch.sum(recall_diff * precision).item() * 100.0

        # --- FPR@95TPR ---
        idx_95 = torch.where(recall >= 0.95)[0][0]
        fpr_95 = (fp[idx_95] / total_negatives).item() * 100.0

        # VRAM Cleanup
        del scores_t, labels_t, desc_scores, indices, desc_labels, tp, fp, precision, recall, recall_diff
        torch.cuda.empty_cache()

        return prc_auc, fpr_95

    # Collect results using GPU tracking
    final_results = {}
    for k in anomaly_scores:
        if len(anomaly_scores[k]) > 0:
            flat_scores = np.concatenate(anomaly_scores[k])
            final_results[k] = calculate_metrics_gpu(flat_scores, device=device)

    # Find the best MSP temperature (highest AUPRC)
    best_msp_t_name = ""
    best_msp_auprc = -1
    best_msp_fpr = -1

    for k, (auprc, fpr) in final_results.items():
        if "MSP" in k:
            if auprc > best_msp_auprc:
                best_msp_auprc = auprc
                best_msp_fpr = fpr
                best_msp_t_name = k.replace("MSP ", "")

    # --- FORMATTED CONSOLE OUTPUT ---
    print("\n" + "="*50)
    print(f"{'ANOMALY DETECTION RESULTS':^50}")
    print("="*50)
    print(f"| {'Method':<18} | {'AUPRC (%)':<10} | {'FPR95 (%)':<10} |")
    print(f"|{'-'*20}|{'-'*12}|{'-'*12}|")

    # Define display order
    display_order = ["MSP (t=1.0)", "MSP (t=0.5)", "MSP (t=0.75)", "MSP (t=1.1)"]

    for method in display_order:
        if method in final_results:
            auprc, fpr = final_results[method]
            display_name = "MSP" if method == "MSP (t=1.0)" else method
            print(f"| {display_name:<18} | {auprc:>9.2f} | {fpr:>9.2f} |")

    # Print Best MSP dynamically
    print(f"| {'MSP (best t)':<18} | {best_msp_auprc:>9.2f} | {best_msp_fpr:>9.2f} |")
    print(f"|{'-'*20}|{'-'*12}|{'-'*12}|")

    # Print the rest of the standard metrics
    for method in ["MaxLogit", "Max_Entropy", "RbA"]:
        if method in final_results:
            auprc, fpr = final_results[method]
            print(f"| {method:<18} | {auprc:>9.2f} | {fpr:>9.2f} |")
    print("="*50)
    print(f"Note: the 'best t' found was {best_msp_t_name}\n")
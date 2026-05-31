# Anomaly Detection Evaluation Script for EOMT
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
    # Nessun Resize! Manteniamo le proporzioni originali (es. 1280x720)
    input_transform = Compose([ToTensor()])

    model.eval()

    ood_gts_list = []
    anomaly_scores = {
        "MSP": [],
        "Max_Entropy": [],
        "MaxLogit": [],
        "RbA": []
    }

    image_paths = glob.glob(os.path.join(images_dir, '*.png'))
    image_paths.extend(glob.glob(os.path.join(images_dir, '*.jpg')))

    print(f"Inizio inferenza su {len(image_paths)} immagini (Risoluzione Naturale)...")

    for img_path in tqdm(image_paths):
        # =============================================================
        # 1. GROUND TRUTH
        # =============================================================
        img_basename = os.path.basename(img_path)
        gt_path = os.path.join(gt_dir, img_basename.replace('.jpg', '.png'))

        if not os.path.exists(gt_path):
            continue

        gt_img = Image.open(gt_path).convert('L')
        gt_array = np.array(gt_img) # Nessun Resize anche qui!

        # Ripristinato il blocco di mappatura originale (0 = Strada, 1 = OOD)
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
        # 2. INFERENCE E WINDOWING
        # =============================================================
        image = Image.open(img_path).convert('RGB')
        img_tensor = input_transform(image)
        img_tensor = (img_tensor * 255).to(torch.uint8)

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            imgs = [img_tensor.to(device)]
            img_sizes = [imgs[0].shape[-2:]]

            crops, origins = model.window_imgs_semantic(imgs)
            mask_logits_per_layer, class_logits_per_layer = model(crops)

            # ---> FIX SPAZIALE: Interpoliamo in base al CROP (es. 512), NON all'immagine globale!
            mask_logits = F.interpolate(mask_logits_per_layer[-1], size=crops.shape[-2:], mode="bilinear", align_corners=False)
            class_logits = class_logits_per_layer[-1]

            # =============================================================
            # 3. STANDARD METRICS (MSP, MaxEntropy, MaxLogit)
            # =============================================================
            crop_probs = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
            probs_list = model.revert_window_logits_semantic(crop_probs, origins, img_sizes)

            # Converti in float32 per precisione numerica (Logit grezzi)
            probs_f32 = probs_list[0].float().cpu().numpy()   # (C, H, W)

            # --- MaxLogit ---
            # Usa i punteggi grezzi
            maxlogit_score = 1.0 - np.max(probs_f32, axis=0)
            anomaly_scores["MaxLogit"].append(maxlogit_score.flatten())

            # --- Softmax Scaling (Base per MSP e MaxEntropy) ---
            e_x = np.exp(probs_f32 - np.max(probs_f32, axis=0, keepdims=True))
            softmax_probs = e_x / np.sum(e_x, axis=0, keepdims=True)

            # --- MSP (Maximum Softmax Probability, t=1.0) ---
            msp_score = 1.0 - np.max(softmax_probs, axis=0)
            anomaly_scores["MSP"].append(msp_score.flatten())

            # --- MaxEntropy ---
            # CALCOLATA SULLE SOFTMAX_PROBS! 
            # Ora i pixel OOD avranno una distribuzione piatta e quindi un'entropia altissima.
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
    # FINAL EVALUATION: AUC AND FPR CALCULATION
    # =============================================================
    print("\nElaborazione dati e calcolo AUC...")
    ood_gts = np.concatenate(ood_gts_list)
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    for k in anomaly_scores:
        if len(anomaly_scores[k]) > 0:
            anomaly_scores[k] = np.concatenate(anomaly_scores[k])

    def eval_metric(scores_array, name):
        val_out = np.concatenate((scores_array[ind_mask], scores_array[ood_mask]))
        val_label = np.concatenate((np.zeros(ind_mask.sum()), np.ones(ood_mask.sum())))

        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)

        print(f"[{name}] AUPRC: {prc_auc*100.0:.2f} | FPR@TPR95: {fpr*100.0:.2f}")

    print("\n" + "="*40)
    print("       RISULTATI ANOMALY DETECTION")
    print("="*40)

    for metric_name, scores_array in anomaly_scores.items():
        if len(scores_array) > 0:
            eval_metric(scores_array, metric_name)
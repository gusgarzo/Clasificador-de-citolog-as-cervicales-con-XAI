"""
eval_architecture.py — XAI analysis of the full fusion architecture
Three analyses:
    1. Side-by-side visualization (original | gradcam | cellpose | prediction)
    2. Branch contribution analysis (which branch drives each decision)
    3. AblationCAM on ConvNeXt
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import datasets, models, transforms
from PIL import Image
from collections import defaultdict

from pytorch_grad_cam import AblationCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils import device, BINARY_MAP
from main import (load_convnext, load_swin, load_cellpose,
                  extract_swin_features, extract_gradcam_map,
                  extract_cellpose_mask, precompute_masks,
                  FusionClassifier, RESNET_PTH, SWIN_PTH, FUSION_PTH,
                  TEST_DIR, GRADCAM_DIM, SEG_DIM, SWIN_DIM)

os.makedirs('results/xai', exist_ok=True)
os.makedirs('results/xai/sidebyside', exist_ok=True)

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════

print("Loading models...")
convnext       = load_convnext(RESNET_PTH)
swin           = load_swin(SWIN_PTH)
cellpose_model = load_cellpose()

fusion = FusionClassifier(swin_dim=SWIN_DIM, gradcam_dim=GRADCAM_DIM, seg_dim=SEG_DIM).to(device)
fusion.load_state_dict(torch.load(FUSION_PTH, map_location=device))
fusion.eval()

cam_extractor = GradCAMPlusPlus(model=convnext, target_layers=[convnext.features[7][-1]])
ablation_cam  = AblationCAM(model=convnext,     target_layers=[convnext.features[7][-1]])

test_dataset = datasets.ImageFolder(root=TEST_DIR)
precompute_masks(test_dataset.imgs, cellpose_model, cache_dir='cache/masks_test')
print("Models loaded.\n")


# ══════════════════════════════════════════════════════════════════════════════
# BRANCH CONTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def branch_contribution(fusion, f_swin, cam_map, seg_mask):
    with torch.no_grad():
        full_prob = torch.softmax(fusion(f_swin, cam_map, seg_mask), dim=1)[0][1].item()

        prob_no_swin = torch.softmax(
            fusion(torch.zeros_like(f_swin), cam_map, seg_mask), dim=1)[0][1].item()
        prob_no_cam  = torch.softmax(
            fusion(f_swin, torch.zeros_like(cam_map), seg_mask), dim=1)[0][1].item()
        prob_no_seg  = torch.softmax(
            fusion(f_swin, cam_map, torch.zeros_like(seg_mask)), dim=1)[0][1].item()

    return {
        'full_prob':            full_prob,
        'swin_contribution':    full_prob - prob_no_swin,
        'gradcam_contribution': full_prob - prob_no_cam,
        'seg_contribution':     full_prob - prob_no_seg,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP — process all test images
# ══════════════════════════════════════════════════════════════════════════════

contributions_by_class = defaultdict(lambda: defaultdict(list))
all_preds, all_labels  = [], []

# how many side-by-side images to save per class
N_EXAMPLES = 10

saved_cancer = 0
saved_benign = 0
saved_wrong  = 0

print("Processing test images...")
for i, (filepath, orig_label) in enumerate(test_dataset.imgs):
    label = BINARY_MAP[orig_label]

    img_pil = Image.open(filepath).convert('RGB')
    tensor  = val_transform(img_pil).unsqueeze(0).to(device)
    rgb     = np.array(img_pil.resize((224, 224))) / 255.0

    # ── Extract features ──
    f_swin = extract_swin_features(swin, tensor)

    with torch.no_grad():
        convnext_pred = convnext(tensor).argmax(1).item()

    cam_map  = extract_gradcam_map(cam_extractor, tensor, convnext_pred).to(device)
    seg_mask = extract_cellpose_mask(i, cache_dir='cache/masks_test').to(device)

    # ── Fusion prediction ──
    with torch.no_grad():
        logits      = fusion(f_swin, cam_map, seg_mask)
        probs       = torch.softmax(logits, dim=1)
        pred        = logits.argmax(1).item()
        cancer_prob = probs[0][1].item()

    all_preds.append(pred)
    all_labels.append(label)

    # ── Branch contribution ──
    contrib = branch_contribution(fusion, f_swin, cam_map, seg_mask)
    class_name = 'cancer' if label == 1 else 'benign'
    for k, v in contrib.items():
        contributions_by_class[class_name][k].append(v)

    # ── GradCAM++ visualization ──
    grayscale_gradcam = cam_extractor(
        input_tensor=tensor,
        targets=[ClassifierOutputTarget(convnext_pred)])[0]
    gradcam_vis = show_cam_on_image(rgb.astype(np.float32), grayscale_gradcam, use_rgb=True)

    # ── AblationCAM visualization ──
    grayscale_ablation = ablation_cam(
        input_tensor=tensor,
        targets=[ClassifierOutputTarget(convnext_pred)])[0]
    ablation_vis = show_cam_on_image(rgb.astype(np.float32), grayscale_ablation, use_rgb=True)

    # ── Cellpose mask visualization ──
    mask_np = seg_mask[0, 0].cpu().numpy()

    # ── Side-by-side figure ──
    correct = (pred == label)
    if (label == 1 and saved_cancer < N_EXAMPLES) or \
       (label == 0 and saved_benign < N_EXAMPLES) or \
       (not correct and saved_wrong < N_EXAMPLES):

        fig = plt.figure(figsize=(18, 4))
        gs  = gridspec.GridSpec(1, 5, figure=fig)

        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(img_pil.resize((224, 224)))
        ax0.set_title('Original')
        ax0.axis('off')

        ax1 = fig.add_subplot(gs[1])
        ax1.imshow(gradcam_vis)
        ax1.set_title('GradCAM++')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[2])
        ax2.imshow(ablation_vis)
        ax2.set_title('AblationCAM')
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[3])
        ax3.imshow(mask_np, cmap='gray')
        ax3.set_title('Cellpose Mask')
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[4])
        contribs = [
            contrib['swin_contribution'],
            contrib['gradcam_contribution'],
            contrib['seg_contribution']
        ]
        colors = ['steelblue', 'coral', 'mediumseagreen']
        ax4.bar(['Swin', 'GradCAM', 'Seg'], contribs, color=colors)
        ax4.set_title(f'Branch contributions\nPred: {"cancer" if pred==1 else "benign"} ({cancer_prob:.2f})')
        ax4.set_ylabel('Contribution to cancer prob')
        ax4.axhline(0, color='black', linewidth=0.8)

        true_str  = 'cancer' if label == 1 else 'benign'
        pred_str  = 'cancer' if pred  == 1 else 'benign'
        correct_str = 'correct' if correct else 'WRONG'
        fig.suptitle(f'True: {true_str} | Pred: {pred_str} | {correct_str}', fontsize=12)

        plt.tight_layout()

        if not correct:
            save_path = f'results/xai/sidebyside/wrong_{saved_wrong:02d}.png'
            saved_wrong += 1
        elif label == 1:
            save_path = f'results/xai/sidebyside/cancer_{saved_cancer:02d}.png'
            saved_cancer += 1
        else:
            save_path = f'results/xai/sidebyside/benign_{saved_benign:02d}.png'
            saved_benign += 1

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(test_dataset.imgs)} processed...")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Average branch contributions per class
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
branch_names = ['Swin', 'GradCAM', 'Seg']
keys = ['swin_contribution', 'gradcam_contribution', 'seg_contribution']
colors = ['steelblue', 'coral', 'mediumseagreen']

for ax, class_name in zip(axes, ['benign', 'cancer']):
    means = [np.mean(contributions_by_class[class_name][k]) for k in keys]
    stds  = [np.std(contributions_by_class[class_name][k])  for k in keys]
    ax.bar(branch_names, means, yerr=stds, color=colors, capsize=5)
    ax.set_title(f'Branch contributions — {class_name}')
    ax.set_ylabel('Avg contribution to cancer prob')
    ax.axhline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig('results/xai/branch_contributions.png', dpi=150)
plt.close()
print("Saved branch_contributions.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Branch contribution scatter: correct vs wrong predictions
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
branch_keys   = ['swin_contribution', 'gradcam_contribution', 'seg_contribution']
branch_labels = ['Swin', 'GradCAM', 'Seg']

for ax, key, bname in zip(axes, branch_keys, branch_labels):
    benign_vals = contributions_by_class['benign'][key]
    cancer_vals = contributions_by_class['cancer'][key]
    ax.hist(benign_vals, bins=20, alpha=0.6, label='benign', color='steelblue')
    ax.hist(cancer_vals, bins=20, alpha=0.6, label='cancer', color='coral')
    ax.set_title(f'{bname} contribution distribution')
    ax.set_xlabel('Contribution to cancer prob')
    ax.set_ylabel('Count')
    ax.legend()

plt.tight_layout()
plt.savefig('results/xai/contribution_distributions.png', dpi=150)
plt.close()
print("Saved contribution_distributions.png")

print("\nAll XAI results saved to results/xai/")
print(f"Side-by-side images: {saved_cancer} cancer, {saved_benign} benign, {saved_wrong} wrong")
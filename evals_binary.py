import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torchvision import datasets, models, transforms
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score,
                             f1_score)
from PIL import Image
from utils import device, BINARY_MAP, CitologiaDataset
from main import (FusionClassifier, extract_swin_features, extract_gradcam_map,
                  extract_cellpose_mask, precompute_masks, load_cellpose,
                  RESNET_PTH, SWIN_PTH, FUSION_PTH, TEST_DIR,
                  GRADCAM_DIM, SEG_DIM, SWIN_DIM)
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

os.makedirs('results/eval', exist_ok=True)

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ── Model loaders ──────────────────────────────────────────────────────────────

def load_convnext(path):
    m = models.convnext_small(weights=None)
    m.classifier[2] = nn.Linear(768, 2)
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    return m

def load_swin(path):
    m = models.swin_t(weights=None)
    m.head = nn.Linear(768, 2)
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    return m


# ── Evaluation functions ───────────────────────────────────────────────────────

def evaluate_model(model, test_loader):
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            preds  = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def evaluate_fusion(fusion, convnext, swin, cam_extractor, test_dataset):
    all_preds, all_labels, all_probs = [], [], []
    for i, (filepath, orig_label) in enumerate(test_dataset.imgs):
        label = BINARY_MAP[orig_label]
        img_pil = Image.open(filepath).convert('RGB')
        tensor  = val_transform(img_pil).unsqueeze(0).to(device)

        f_swin = extract_swin_features(swin, tensor)
        with torch.no_grad():
            convnext_pred = convnext(tensor).argmax(1).item()
        cam_map  = extract_gradcam_map(cam_extractor, tensor, convnext_pred).to(device)
        seg_mask = extract_cellpose_mask(i, cache_dir='cache/masks_test').to(device)

        with torch.no_grad():
            logits      = fusion(f_swin, cam_map, seg_mask)
            probs       = torch.softmax(logits, dim=1)
            pred        = logits.argmax(1).item()
            cancer_prob = probs[0][1].item()

        all_preds.append(pred)
        all_labels.append(label)
        all_probs.append(cancer_prob)

        if (i + 1) % 20 == 0:
            print(f"  Fusion eval: {i+1}/{len(test_dataset.imgs)}...")

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ── Dataset ────────────────────────────────────────────────────────────────────

test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=val_transform)
test_binary  = CitologiaDataset(test_dataset, BINARY_MAP)
test_loader  = torch.utils.data.DataLoader(test_binary, batch_size=32, shuffle=False)


# ── Load models ────────────────────────────────────────────────────────────────

print("Loading models...")
convnext = load_convnext(RESNET_PTH)
swin_cls = load_swin(SWIN_PTH)
cellpose_model = load_cellpose()

swin_feat = load_swin(SWIN_PTH)
cam_extractor = GradCAMPlusPlus(model=convnext, target_layers=[convnext.features[7][-1]])

fusion = FusionClassifier(swin_dim=SWIN_DIM, gradcam_dim=GRADCAM_DIM, seg_dim=SEG_DIM).to(device)
fusion.load_state_dict(torch.load(FUSION_PTH, map_location=device))
fusion.eval()

precompute_masks(test_dataset.imgs, cellpose_model, cache_dir='cache/masks_test')


# ── Run evaluations ────────────────────────────────────────────────────────────

print("\nEvaluating ConvNeXt...")
labels_cx, preds_cx, probs_cx = evaluate_model(convnext, test_loader)

print("Evaluating Swin-T...")
labels_sw, preds_sw, probs_sw = evaluate_model(swin_cls, test_loader)

print("Evaluating Fusion...")
labels_fu, preds_fu, probs_fu = evaluate_fusion(fusion, convnext, swin_feat, cam_extractor, test_dataset)


# ── Print classification reports ───────────────────────────────────────────────

print("\n=== ConvNeXt-Small ===")
print(classification_report(labels_cx, preds_cx, target_names=['benign', 'cancer']))
print(f"AUC: {roc_auc_score(labels_cx, probs_cx):.3f}")
print(f"AP:  {average_precision_score(labels_cx, probs_cx):.3f}")

print("\n=== Swin-T ===")
print(classification_report(labels_sw, preds_sw, target_names=['benign', 'cancer']))
print(f"AUC: {roc_auc_score(labels_sw, probs_sw):.3f}")
print(f"AP:  {average_precision_score(labels_sw, probs_sw):.3f}")

print("\n=== Fusion ===")
print(classification_report(labels_fu, preds_fu, target_names=['benign', 'cancer']))
print(f"AUC: {roc_auc_score(labels_fu, probs_fu):.3f}")
print(f"AP:  {average_precision_score(labels_fu, probs_fu):.3f}")

# Save to text file
with open('results/eval/metrics.txt', 'w') as f:
    f.write("=== ConvNeXt-Small ===\n")
    f.write(classification_report(labels_cx, preds_cx, target_names=['benign', 'cancer']))
    f.write(f"AUC: {roc_auc_score(labels_cx, probs_cx):.3f}\n")
    f.write(f"AP:  {average_precision_score(labels_cx, probs_cx):.3f}\n\n")
    f.write("=== Swin-T ===\n")
    f.write(classification_report(labels_sw, preds_sw, target_names=['benign', 'cancer']))
    f.write(f"AUC: {roc_auc_score(labels_sw, probs_sw):.3f}\n")
    f.write(f"AP:  {average_precision_score(labels_sw, probs_sw):.3f}\n\n")
    f.write("=== Fusion ===\n")
    f.write(classification_report(labels_fu, preds_fu, target_names=['benign', 'cancer']))
    f.write(f"AUC: {roc_auc_score(labels_fu, probs_fu):.3f}\n")
    f.write(f"AP:  {average_precision_score(labels_fu, probs_fu):.3f}\n")
print("Saved metrics.txt")


# ── Figure 1: Confusion matrices (3 models) ───────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, labels, preds, title in zip(
    axes,
    [labels_cx,       labels_sw,  labels_fu],
    [preds_cx,        preds_sw,   preds_fu],
    ['ConvNeXt-Small','Swin-T',   'Fusion']
):
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['benign', 'cancer'],
                yticklabels=['benign', 'cancer'])
    ax.set(xlabel='Predicted', ylabel='True', title=title)

plt.tight_layout()
plt.savefig('results/eval/confusion_matrices.png', dpi=150)
plt.close()
print("Saved confusion_matrices.png")


# ── Figure 2: ROC curves (3 models overlaid) ──────────────────────────────────

plt.figure(figsize=(6, 5))
for labels, probs, name, color in [
    (labels_cx, probs_cx, 'ConvNeXt-Small', '#185FA5'),
    (labels_sw, probs_sw, 'Swin-T',         '#D85A30'),
    (labels_fu, probs_fu, 'Fusion',          '#1D9E75'),
]:
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', color=color, lw=2)

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('results/eval/roc_curves.png', dpi=150)
plt.close()
print("Saved roc_curves.png")


# ── Figure 3: Precision-Recall curves (3 models overlaid) ─────────────────────

plt.figure(figsize=(6, 5))
for labels, probs, name, color in [
    (labels_cx, probs_cx, 'ConvNeXt-Small', '#185FA5'),
    (labels_sw, probs_sw, 'Swin-T',         '#D85A30'),
    (labels_fu, probs_fu, 'Fusion',          '#1D9E75'),
]:
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    plt.plot(recall, precision, label=f'{name} (AP = {ap:.3f})', color=color, lw=2)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('results/eval/pr_curves.png', dpi=150)
plt.close()
print("Saved pr_curves.png")


# ── Figure 4: Bar chart comparison (3 models) ─────────────────────────────────

metrics = {
    'F1 Cancer': [
        f1_score(labels_cx, preds_cx, pos_label=1),
        f1_score(labels_sw, preds_sw, pos_label=1),
        f1_score(labels_fu, preds_fu, pos_label=1),
    ],
    'F1 Benign': [
        f1_score(labels_cx, preds_cx, pos_label=0),
        f1_score(labels_sw, preds_sw, pos_label=0),
        f1_score(labels_fu, preds_fu, pos_label=0),
    ],
    'AUC': [
        roc_auc_score(labels_cx, probs_cx),
        roc_auc_score(labels_sw, probs_sw),
        roc_auc_score(labels_fu, probs_fu),
    ],
    'AP': [
        average_precision_score(labels_cx, probs_cx),
        average_precision_score(labels_sw, probs_sw),
        average_precision_score(labels_fu, probs_fu),
    ],
}

x      = np.arange(len(metrics))
width  = 0.25
names  = list(metrics.keys())
colors = ['#185FA5', '#D85A30', '#1D9E75']
labels_list = ['ConvNeXt-Small', 'Swin-T', 'Fusion']

fig, ax = plt.subplots(figsize=(9, 5))
for i, (color, label) in enumerate(zip(colors, labels_list)):
    vals = [v[i] for v in metrics.values()]
    ax.bar(x + (i - 1) * width, vals, width, label=label, color=color)

ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0.8, 1.02)
ax.set_ylabel('Score')
ax.set_title('Model Comparison')
ax.legend()
plt.tight_layout()
plt.savefig('results/eval/comparison_bar.png', dpi=150)
plt.close()
print("Saved comparison_bar.png")

print("\nAll evaluation results saved to results/eval/")
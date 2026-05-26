import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torchvision import datasets, models, transforms
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score)
from PIL import Image
from utils import device, BINARY_MAP, CitologiaDataset, TARGET_NAMES
from main import (FusionClassifier, extract_swin_features, extract_gradcam_map,
                  extract_cellpose_mask, precompute_masks, load_cellpose,
                  TEST_DIR, GRADCAM_DIM, SEG_DIM, SWIN_DIM)
from pytorch_grad_cam import GradCAMPlusPlus

os.makedirs('results/eval_4class', exist_ok=True)

RESNET_PTH_4 = 'best_model_4class.pth'
SWIN_PTH_4   = 'best_swin_4class.pth'
FUSION_PTH_4 = 'best_fusion_4class.pth'

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ── Model loaders ──────────────────────────────────────────────────────────────

def load_convnext_4(path):
    m = models.convnext_small(weights=None)
    m.classifier[2] = nn.Linear(768, 4)
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    return m

def load_swin_4(path):
    m = models.swin_t(weights=None)
    m.head = nn.Linear(768, 4)
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
            probs  = torch.softmax(logits, dim=1)
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
            logits = fusion(f_swin, cam_map, seg_mask)
            probs  = torch.softmax(logits, dim=1)
            pred   = logits.argmax(1).item()

        all_preds.append(pred)
        all_labels.append(label)
        all_probs.append(probs[0].cpu().numpy())

        if (i + 1) % 20 == 0:
            print(f"  Fusion eval: {i+1}/{len(test_dataset.imgs)}...")

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ── Dataset ────────────────────────────────────────────────────────────────────

test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=val_transform)
test_binary  = CitologiaDataset(test_dataset, BINARY_MAP)
test_loader  = torch.utils.data.DataLoader(test_binary, batch_size=32, shuffle=False)


# ── Load models ────────────────────────────────────────────────────────────────

print("Loading models...")
convnext = load_convnext_4(RESNET_PTH_4)
swin_cls = load_swin_4(SWIN_PTH_4)
cellpose_model = load_cellpose()

swin_feat     = load_swin_4(SWIN_PTH_4)
cam_extractor = GradCAMPlusPlus(model=convnext, target_layers=[convnext.features[7][-1]])

fusion = FusionClassifier(swin_dim=SWIN_DIM, gradcam_dim=GRADCAM_DIM, seg_dim=SEG_DIM).to(device)
fusion.load_state_dict(torch.load(FUSION_PTH_4, map_location=device))
fusion.eval()

precompute_masks(test_dataset.imgs, cellpose_model, cache_dir='cache/masks_test')


# ── Run evaluations ────────────────────────────────────────────────────────────

print("\nEvaluating ConvNeXt...")
labels_cx, preds_cx, probs_cx = evaluate_model(convnext, test_loader)

print("Evaluating Swin-T...")
labels_sw, preds_sw, probs_sw = evaluate_model(swin_cls, test_loader)

print("Evaluating Fusion...")
labels_fu, preds_fu, probs_fu = evaluate_fusion(fusion, convnext, swin_feat, cam_extractor, test_dataset)


# ── Print and save classification reports ─────────────────────────────────────

with open('results/eval_4class/metrics.txt', 'w') as f:
    for name, labels, preds, probs in [
        ('ConvNeXt-Small', labels_cx, preds_cx, probs_cx),
        ('Swin-T',         labels_sw, preds_sw, probs_sw),
        ('Fusion',         labels_fu, preds_fu, probs_fu),
    ]:
        report = classification_report(labels, preds, target_names=TARGET_NAMES)
        auc    = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
        macro_f1 = f1_score(labels, preds, average='macro')

        print(f"\n=== {name} ===")
        print(report)
        print(f"Macro F1: {macro_f1:.3f}")
        print(f"AUC (OvR macro): {auc:.3f}")

        f.write(f"=== {name} ===\n")
        f.write(report)
        f.write(f"Macro F1: {macro_f1:.3f}\n")
        f.write(f"AUC (OvR macro): {auc:.3f}\n\n")

print("Saved metrics.txt")


# ── Figure 1: Confusion matrices ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, labels, preds, title in zip(
    axes,
    [labels_cx,        labels_sw,  labels_fu],
    [preds_cx,         preds_sw,   preds_fu],
    ['ConvNeXt-Small', 'Swin-T',   'Fusion']
):
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=TARGET_NAMES,
                yticklabels=TARGET_NAMES)
    ax.set(xlabel='Predicted', ylabel='True', title=title)

plt.tight_layout()
plt.savefig('results/eval_4class/confusion_matrices.png', dpi=150)
plt.close()
print("Saved confusion_matrices.png")


# ── Figure 2: Macro F1 bar chart ──────────────────────────────────────────────

model_names = ['ConvNeXt-Small', 'Swin-T', 'Fusion']
colors      = ['#185FA5', '#D85A30', '#1D9E75']

per_class_f1 = {}
for name, labels, preds in [
    ('ConvNeXt-Small', labels_cx, preds_cx),
    ('Swin-T',         labels_sw, preds_sw),
    ('Fusion',         labels_fu, preds_fu),
]:
    scores = f1_score(labels, preds, average=None)
    per_class_f1[name] = scores

x     = np.arange(len(TARGET_NAMES))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
for i, (name, color) in enumerate(zip(model_names, colors)):
    ax.bar(x + (i - 1) * width, per_class_f1[name], width, label=name, color=color)

ax.set_xticks(x)
ax.set_xticklabels(TARGET_NAMES)
ax.set_ylim(0, 1.05)
ax.set_ylabel('F1 Score')
ax.set_title('Per-class F1 Comparison')
ax.legend()
plt.tight_layout()
plt.savefig('results/eval_4class/f1_comparison.png', dpi=150)
plt.close()
print("Saved f1_comparison.png")


# ── Figure 3: AUC bar chart ───────────────────────────────────────────────────

auc_scores = []
for labels, probs in [(labels_cx, probs_cx), (labels_sw, probs_sw), (labels_fu, probs_fu)]:
    auc_scores.append(roc_auc_score(labels, probs, multi_class='ovr', average='macro'))

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(model_names, auc_scores, color=colors, width=0.4)
ax.set_ylim(0.5, 1.05)
ax.set_ylabel('AUC (OvR macro)')
ax.set_title('AUC Comparison — 4-class')
for bar, val in zip(bars, auc_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('results/eval_4class/auc_comparison.png', dpi=150)
plt.close()
print("Saved auc_comparison.png")

print("\nAll 4-class evaluation results saved to results/eval_4class/")
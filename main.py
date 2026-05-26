"""
main.py — Pipeline de fusión multimodal
Ramas: Swin-T (features semánticas) + ResNet50 GradCAM++ (saliencia) + Cellpose (segmentación)

Uso:
    python main.py --mode train
    python main.py --mode eval
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score)
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from cellpose import models as cp_models
from utils import device, BINARY_MAP, CitologiaDataset, TARGET_NAMES


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_DIR   = 'dataset/entrenamiento'
TEST_DIR    = 'dataset/test'
#RESNET_PTH  = 'best_model.pth'
#SWIN_PTH    = 'best_swin.pth'
#FUSION_PTH  = 'best_fusion.pth'
RESNET_PTH  = 'best_model_4class.pth'
SWIN_PTH    = 'best_swin_4class.pth'
FUSION_PTH  = 'best_fusion_4class.pth'

EPOCHS      = 100
BATCH       = 16        # más bajo porque cada muestra corre 3 ramas
NUM_WORKERS = 4         # 4 en idun
GRADCAM_DIM = 128       # features extraídas del mapa de calor
SEG_DIM     = 128       # features extraídas de la máscara
SWIN_DIM    = 768       # dimensión del feature map de Swin-T





# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=180),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE MODELOS BASE (congelados — solo extraen features)
# ══════════════════════════════════════════════════════════════════════════════

def load_convnext(path):
    m = models.convnext_small(weights=None)
    #m.classifier[2] = nn.Linear(768, 2)
    m.classifier[2] = nn.Linear(768, 4) 
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    return m


def load_swin(path):
    """Swin-T entrenado — extrae feature map global de 768d."""
    m = models.swin_t(weights=None)
    #m.head = nn.Linear(768, 2)
    m.head = nn.Linear(768, 4)
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def load_cellpose():
    return cp_models.CellposeModel(gpu=(device.type == 'cuda'), model_type='cpsam')

# ══════════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def extract_swin_features(swin, tensor):
    """
    tensor: (B, 3, 224, 224) ya en device
    Devuelve: (B, 768) — average pooling sobre el feature map final
    """
    with torch.no_grad():
        x = swin.features(tensor)   # (B, 7, 7, 768)
        x = swin.norm(x)
        x = x.mean(dim=[1, 2])     # (B, 768)
    return x


def extract_gradcam_map(cam_extractor, tensor, label):
    """
    tensor: (1, 3, 224, 224) en device
    label:  int — clase objetivo para GradCAM++
    Devuelve: (1, 1, 224, 224) tensor float normalizado [0,1]
    """
    targets = [ClassifierOutputTarget(label)]
    grayscale = cam_extractor(input_tensor=tensor, targets=targets)  # (1, 224, 224)
    cam_tensor = torch.tensor(grayscale, dtype=torch.float32).unsqueeze(1)  # (1,1,224,224)
    return cam_tensor

def precompute_masks(img_paths, cellpose_model, cache_dir='cache/masks'):
    os.makedirs(cache_dir, exist_ok=True)
    print("Precomputing Cellpose masks...")
    for idx, (filepath, _) in enumerate(img_paths):
        cache_path = f'{cache_dir}/{idx}.npy'
        if os.path.exists(cache_path):
            continue
        img = np.array(Image.open(filepath).convert('RGB').resize((224, 224)))
        masks, _, _ = cellpose_model.eval(img, diameter=30)  # try 30, 50, or 80
        binary_mask = (masks > 0).astype(np.float32)
        np.save(cache_path, binary_mask)
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(img_paths)} done...")
    print("Masks cached.")
def extract_cellpose_mask(idx, cache_dir='cache/masks'):
    binary_mask = np.load(f'{cache_dir}/{idx}.npy')
    return torch.tensor(binary_mask).unsqueeze(0).unsqueeze(0)  # (1,1,224,224)

def precompute_swin_features(img_paths, swin, cache_dir='cache/swin'):
    os.makedirs(cache_dir, exist_ok=True)
    print("Precomputing Swin features...")
    for idx, (filepath, _) in enumerate(img_paths):
        cache_path = f'{cache_dir}/{idx}.npy'
        if os.path.exists(cache_path):
            continue
        img_pil = Image.open(filepath).convert('RGB')
        tensor = val_transform(img_pil).unsqueeze(0).to(device)
        feat = extract_swin_features(swin, tensor)
        np.save(cache_path, feat.cpu().numpy())
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(img_paths)} done...")
    print("Swin features cached.")

def precompute_gradcam_maps(img_paths, convnext, cam_extractor, cache_dir='cache/gradcam'):
    os.makedirs(cache_dir, exist_ok=True)
    print("Precomputing GradCAM maps...")
    for idx, (filepath, _) in enumerate(img_paths):
        cache_path = f'{cache_dir}/{idx}.npy'
        if os.path.exists(cache_path):
            continue
        img_pil = Image.open(filepath).convert('RGB')
        tensor = val_transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = convnext(tensor).argmax(1).item()
        cam = extract_gradcam_map(cam_extractor, tensor, pred)
        np.save(cache_path, cam.numpy())
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(img_paths)} done...")
    print("GradCAM maps cached.")
# ══════════════════════════════════════════════════════════════════════════════
# CNNs LIGERAS para mapa de calor y máscara
# ══════════════════════════════════════════════════════════════════════════════

class LightCNN(nn.Module):
    """
    Procesa un mapa de 1 canal (224x224) y extrae un vector de features.
    Usada tanto para el mapa GradCAM++ como para la máscara de segmentación.
    """
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),  # 112
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), # 56
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), # 28
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # (B, 128, 1, 1)
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# CLASIFICADOR DE FUSIÓN
# ══════════════════════════════════════════════════════════════════════════════

class FusionClassifier(nn.Module):
    """
    Concatena features de las tres ramas y clasifica con un MLP.
    Dimensiones:
        Swin:    768
        GradCAM: GRADCAM_DIM (128)
        Seg:     SEG_DIM     (128)
        Total:   1024
    """
    def __init__(self, swin_dim=SWIN_DIM, gradcam_dim=GRADCAM_DIM, seg_dim=SEG_DIM):
        super().__init__()
        self.gradcam_cnn = LightCNN(out_dim=gradcam_dim)
        self.seg_cnn     = LightCNN(out_dim=seg_dim)

        total = swin_dim + gradcam_dim + seg_dim  # 1024

        self.mlp = nn.Sequential(
            nn.Linear(total, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128),  nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 4)       #2 O 4 dependiendo si queremos clasificacion binaria o cuaternaria
        )

    def forward(self, f_swin, cam_map, seg_mask):
        """
        f_swin:   (B, 768)       — ya extraído y congelado
        cam_map:  (B, 1, 224, 224)
        seg_mask: (B, 1, 224, 224)
        """
        f_cam = self.gradcam_cnn(cam_map)   # (B, 128)
        f_seg = self.seg_cnn(seg_mask)      # (B, 128)
        x = torch.cat([f_swin, f_cam, f_seg], dim=1)  # (B, 1024)
        return self.mlp(x)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE — construye tensores de las 3 ramas para un batch de índices
# ══════════════════════════════════════════════════════════════════════════════

def build_batch(indices, img_paths, binary_map):
    f_swin_list, cam_list, seg_list, label_list = [], [], [], []

    for idx in indices:
        filepath, orig_label = img_paths[idx]
        label = binary_map[orig_label]

        f_swin   = torch.tensor(np.load(f'cache/swin/{idx}.npy'))
        cam_map  = torch.tensor(np.load(f'cache/gradcam/{idx}.npy'))
        seg_mask = torch.tensor(np.load(f'cache/masks/{idx}.npy')).unsqueeze(0).unsqueeze(0)

        f_swin_list.append(f_swin)
        cam_list.append(cam_map)
        seg_list.append(seg_mask)
        label_list.append(label)

    f_swin_batch = torch.cat(f_swin_list, dim=0).to(device)
    cam_batch    = torch.cat(cam_list,    dim=0).to(device)
    seg_batch    = torch.cat(seg_list,    dim=0).to(device)
    labels_batch = torch.tensor(label_list, dtype=torch.long).to(device)

    return f_swin_batch, cam_batch, seg_batch, labels_batch

# ══════════════════════════════════════════════════════════════════════════════
# MODO TRAIN, aplicar dropout
# ══════════════════════════════════════════════════════════════════════════════
def apply_modality_dropout(f_swin, cam_map, seg_mask, p=0.3):
    """
    During training, randomly zero out entire branches with probability p.
    Forces the MLP to not rely exclusively on Swin.
    """
    if torch.rand(1).item() < p:
        f_swin = torch.zeros_like(f_swin)
    if torch.rand(1).item() < p:
        cam_map = torch.zeros_like(cam_map)
    if torch.rand(1).item() < p:
        seg_mask = torch.zeros_like(seg_mask)
    return f_swin, cam_map, seg_mask


def train(args):
    print("\n=== MODO ENTRENAMIENTO ===\n")

    # Carga modelos base
    convnext = load_convnext(RESNET_PTH)
    swin    = load_swin(SWIN_PTH)
    cellpose_model = load_cellpose()
    cam_extractor = GradCAMPlusPlus(model=convnext, target_layers=[convnext.features[7][-1]])

    

    # Dataset base (sin transform aún — lo aplicamos en build_batch)
    raw_dataset = datasets.ImageFolder(root=TRAIN_DIR)
    precompute_masks(raw_dataset.imgs, cellpose_model)
    precompute_swin_features(raw_dataset.imgs, swin)
    precompute_gradcam_maps(raw_dataset.imgs, convnext, cam_extractor)
    targets = [BINARY_MAP[label] for _, label in raw_dataset.imgs]

    train_idx, val_idx = train_test_split(
        range(len(raw_dataset)),
        test_size=0.2,
        stratify=targets,
        random_state=42
    )   
    train_idx = list(train_idx)
    val_idx   = list(val_idx)

    # Pesos de clase para loss
    counts = Counter(targets)
    total_samples = sum(counts.values())
    weights = [total_samples / (4 * counts[i]) for i in range(4)]
    weight_tensor = torch.tensor(weights).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # Modelo de fusión — solo este se entrena
    fusion = FusionClassifier().to(device)
    optimizer = optim.AdamW(fusion.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_f1 = 0.0

    for epoch in range(EPOCHS):
        fusion.train()
        loss_acum = 0.0

        # Iterar en mini-batches sobre los índices de entrenamiento
        np.random.shuffle(train_idx)
        batches = [train_idx[i:i+BATCH] for i in range(0, len(train_idx), BATCH)]
        for batch_indices in batches:
            f_swin, cam_map, seg_mask, labels = build_batch(batch_indices, raw_dataset.imgs, BINARY_MAP)
            #f_swin, cam_map, seg_mask = apply_modality_dropout(f_swin, cam_map, seg_mask)
            optimizer.zero_grad()
            logits = fusion(f_swin, cam_map, seg_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), max_norm=1.0)
            optimizer.step()
            loss_acum += loss.item()

        scheduler.step()

        # Validación
        fusion.eval()
        preds_list, labels_list = [], []

        val_batches = [val_idx[i:i+BATCH] for i in range(0, len(val_idx), BATCH)]
        with torch.no_grad():
            for batch_indices in val_batches:
                f_swin, cam_map, seg_mask, labels = build_batch(batch_indices, raw_dataset.imgs, BINARY_MAP)
                logits = fusion(f_swin, cam_map, seg_mask)
                preds_list.append(logits.argmax(1).cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        preds_flat  = np.concatenate(preds_list)
        labels_flat = np.concatenate(labels_list)
        #f1 = f1_score(labels_flat, preds_flat, pos_label=1)
        
        f1 = f1_score(labels_flat, preds_flat, average='macro')
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss_acum/len(batches):.4f} | Val F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(fusion.state_dict(), FUSION_PTH)
            print(f"  ✓ Nuevo mejor F1: {best_f1:.4f} — guardado en {FUSION_PTH}")

    print(f"\nEntrenamiento completado. Mejor F1: {best_f1:.4f}")
    print(classification_report(labels_flat, preds_flat, target_names=TARGET_NAMES))


# ══════════════════════════════════════════════════════════════════════════════
# MODO EVAL
# ══════════════════════════════════════════════════════════════════════════════
'''
def evaluate(args):
    print("\n=== MODO EVALUACIÓN ===\n")

    os.makedirs('results/correct_cancer', exist_ok=True)
    os.makedirs('results/correct_benign', exist_ok=True)
    os.makedirs('results/wrong',          exist_ok=True)

    # Carga modelos
    convnext = load_convnext(RESNET_PTH)
    swin    = load_swin(SWIN_PTH)
    cellpose_model = load_cellpose()
    
    cam_extractor = GradCAMPlusPlus(model=convnext, target_layers=[convnext.features[7][-1]])

    fusion = FusionClassifier().to(device)
    fusion.load_state_dict(torch.load(FUSION_PTH, map_location=device))
    fusion.eval()

    test_dataset = datasets.ImageFolder(root=TEST_DIR)
    precompute_masks(test_dataset.imgs, cellpose_model, cache_dir='cache/masks_test')
    all_preds, all_labels, all_probs = [], [], []

    for i, (filepath, orig_label) in enumerate(test_dataset.imgs):
        label = BINARY_MAP[orig_label]

        img_pil = Image.open(filepath).convert('RGB')
        tensor  = val_transform(img_pil).unsqueeze(0).to(device)

        # Features de las 3 ramas
        f_swin   = extract_swin_features(swin, tensor)
        with torch.no_grad():
            convnext_pred = convnext(tensor).argmax(1).item()
        cam_map = extract_gradcam_map(cam_extractor, tensor, convnext_pred).to(device)
        seg_mask = extract_cellpose_mask(i, cache_dir='cache/masks_test').to(device)

        with torch.no_grad():
            logits = fusion(f_swin, cam_map, seg_mask)
            probs  = torch.softmax(logits, dim=1)
            pred   = logits.argmax(1).item()
            cancer_prob = probs[0][1].item()

        all_preds.append(pred)
        all_labels.append(label)
        all_probs.append(cancer_prob)

        # Guarda visualización GradCAM
        rgb = np.array(img_pil.resize((224, 224))) / 255.0
        from pytorch_grad_cam.utils.image import show_cam_on_image
        grayscale = cam_extractor(input_tensor=tensor,
                          targets=[ClassifierOutputTarget(convnext_pred)])[0]
        vis = show_cam_on_image(rgb.astype(np.float32), grayscale, use_rgb=True)

        filename = f'image_{i:04d}_p{cancer_prob:.2f}.png'
        if pred == label == 1:
            plt.imsave(f'results/correct_cancer/{filename}', vis)
        elif pred == label == 0:
            plt.imsave(f'results/correct_benign/{filename}', vis)
        else:
            plt.imsave(f'results/wrong/{filename}', vis)

        if (i + 1) % 10 == 0:
            print(f"  Procesadas {i+1}/{len(test_dataset.imgs)} imágenes...")

    # ── Métricas ──
    print("\n" + classification_report(all_labels, all_preds, target_names=TARGET_NAMES))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print(f"AUC: {roc_auc_score(all_labels, all_probs):.3f}")
    print(f"AP:  {average_precision_score(all_labels, all_probs):.3f}")

    with open('results/metrics.txt', 'w') as f:
        f.write("=== FUSION MODEL — TEST SET RESULTS ===\n\n")
        f.write(classification_report(all_labels, all_preds, target_names=TARGET_NAMES))
        f.write(f"\nAUC: {roc_auc_score(all_labels, all_probs):.3f}\n")
        f.write(f"AP:  {average_precision_score(all_labels, all_probs):.3f}\n")
        cm = confusion_matrix(all_labels, all_preds)
        f.write(f"\nConfusion Matrix:\n{cm}\n")

    # ── Gráficas ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ROC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    axes[0].plot(fpr, tpr, label=f'AUC = {roc_auc_score(all_labels, all_probs):.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set(xlabel='FPR', ylabel='TPR', title='ROC Curve')
    axes[0].legend()

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=TARGET_NAMES,
                yticklabels=TARGET_NAMES)
    axes[1].set(xlabel='Predicted', ylabel='True', title='Confusion Matrix')

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    axes[2].plot(recall, precision, label=f'AP = {ap:.3f}')
    axes[2].set(xlabel='Recall', ylabel='Precision', title='Precision-Recall')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('results/metrics.png', dpi=150)
    plt.show()
    print("\nGráficas guardadas en results/metrics.png")
'''
def evaluate(args):
    print("\n=== MODO EVALUACIÓN ===\n")

    os.makedirs('results/eval_4class', exist_ok=True)

    # Carga modelos
    convnext = load_convnext(RESNET_PTH)
    swin     = load_swin(SWIN_PTH)
    cellpose_model = load_cellpose()

    cam_extractor = GradCAMPlusPlus(model=convnext, target_layers=[convnext.features[7][-1]])

    fusion = FusionClassifier().to(device)
    fusion.load_state_dict(torch.load(FUSION_PTH, map_location=device))
    fusion.eval()

    test_dataset = datasets.ImageFolder(root=TEST_DIR)
    precompute_masks(test_dataset.imgs, cellpose_model, cache_dir='cache/masks_test')
    all_preds, all_labels, all_probs = [], [], []

    # one folder per class + wrong
    for name in TARGET_NAMES:
        os.makedirs(f'results/eval_4class/correct_{name}', exist_ok=True)
    os.makedirs('results/eval_4class/wrong', exist_ok=True)

    for i, (filepath, orig_label) in enumerate(test_dataset.imgs):
        label = BINARY_MAP[orig_label]

        img_pil = Image.open(filepath).convert('RGB')
        tensor  = val_transform(img_pil).unsqueeze(0).to(device)

        # Features de las 3 ramas
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

        # Guarda visualización GradCAM
        rgb = np.array(img_pil.resize((224, 224))) / 255.0
        from pytorch_grad_cam.utils.image import show_cam_on_image
        grayscale = cam_extractor(input_tensor=tensor,
                      targets=[ClassifierOutputTarget(convnext_pred)])[0]
        vis = show_cam_on_image(rgb.astype(np.float32), grayscale, use_rgb=True)

        filename = f'image_{i:04d}_true{TARGET_NAMES[label]}_pred{TARGET_NAMES[pred]}.png'
        if pred == label:
            plt.imsave(f'results/eval_4class/correct_{TARGET_NAMES[label]}/{filename}', vis)
        else:
            plt.imsave(f'results/eval_4class/wrong/{filename}', vis)

        if (i + 1) % 10 == 0:
            print(f"  Procesadas {i+1}/{len(test_dataset.imgs)} imagenes...")

    all_probs = np.array(all_probs)

    # ── Metricas ──
    report = classification_report(all_labels, all_preds, target_names=TARGET_NAMES)
    auc    = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print("\n" + report)
    print(f"Macro F1: {macro_f1:.3f}")
    print(f"AUC (OvR macro): {auc:.3f}")

    with open('results/eval_4class/metrics.txt', 'w') as f:
        f.write("=== FUSION MODEL — 4-CLASS TEST SET RESULTS ===\n\n")
        f.write(report)
        f.write(f"\nMacro F1: {macro_f1:.3f}\n")
        f.write(f"AUC (OvR macro): {auc:.3f}\n")
        cm = confusion_matrix(all_labels, all_preds)
        f.write(f"\nConfusion Matrix:\n{cm}\n")

    # ── Confusion matrix ──
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=TARGET_NAMES,
                yticklabels=TARGET_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix — Fusion 4-class')
    plt.tight_layout()
    plt.savefig('results/eval_4class/confusion_matrix_fusion.png', dpi=150)
    plt.close()

    # ── Per-class F1 bar chart ──
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    plt.figure(figsize=(6, 4))
    bars = plt.bar(TARGET_NAMES, per_class_f1, color=['#1D9E75']*4)
    plt.ylim(0, 1.05)
    plt.ylabel('F1 Score')
    plt.title('Per-class F1 — Fusion 4-class')
    for bar, val in zip(bars, per_class_f1):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('results/eval_4class/f1_fusion.png', dpi=150)
    plt.close()

    print("\nResultados guardados en results/eval_4class/")

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline de fusión multimodal')
    parser.add_argument('--mode', choices=['train', 'eval'], required=True,
                        help='train: entrena el MLP de fusión | eval: evalúa sobre test set')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)
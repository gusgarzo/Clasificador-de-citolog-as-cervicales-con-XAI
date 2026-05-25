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


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_DIR   = 'dataset/entrenamiento'
TEST_DIR    = 'dataset/test'
RESNET_PTH  = 'best_model.pth'
SWIN_PTH    = 'best_swin.pth'
FUSION_PTH  = 'best_fusion.pth'

EPOCHS      = 30
BATCH       = 16        # más bajo porque cada muestra corre 3 ramas
NUM_WORKERS = 0         # 4 en idun
GRADCAM_DIM = 128       # features extraídas del mapa de calor
SEG_DIM     = 128       # features extraídas de la máscara
SWIN_DIM    = 768       # dimensión del feature map de Swin-T
BINARY_MAP  = {0: 1, 1: 1, 2: 1, 3: 0}   # benigna=0, resto=1

device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps'  if torch.backends.mps.is_available() else
    'cpu'
)
print(f"Device: {device}")


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class CitologiaDataset(data.Dataset):
    def __init__(self, original_dataset, binary_map):
        self.original_dataset = original_dataset
        self.binary_map = binary_map

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        img, label = self.original_dataset[idx]
        return img, self.binary_map[label]


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

def load_resnet(path):
    """ResNet50 entrenado — solo para GradCAM++, no se entrena más."""
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(2048, 2)
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    return m


def load_swin(path):
    """Swin-T entrenado — extrae feature map global de 768d."""
    m = models.swin_t(weights=None)
    m.head = nn.Linear(768, 2)
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def load_cellpose():
    """Cellpose preentrenado en núcleos — no necesita entrenamiento."""
    return cp_models.CellposeModel(gpu=(device.type == 'cuda'), model_type='nuclei')


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


def extract_cellpose_mask(cellpose_model, filepath):
    """
    filepath: ruta a la imagen original
    Devuelve: (1, 1, 224, 224) tensor float con máscara binaria de núcleos
    """
    img = np.array(Image.open(filepath).convert('RGB').resize((224, 224)))
    masks, _, _ = cellpose_model.eval(img, diameter=None)
    binary_mask = (masks > 0).astype(np.float32)
    mask_tensor = torch.tensor(binary_mask).unsqueeze(0).unsqueeze(0)  # (1,1,224,224)
    return mask_tensor


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
            nn.Linear(128, 2)
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

def build_batch(indices, base_dataset, img_paths, swin, cam_extractor, cellpose_model,
                transform, binary_map):
    """
    Construye manualmente un batch completo para los índices dados.
    Necesario porque Cellpose opera sobre PIL images, no tensores.

    Devuelve:
        f_swin_batch:   (B, 768)
        cam_batch:      (B, 1, 224, 224)
        seg_batch:      (B, 1, 224, 224)
        labels_batch:   (B,)
    """
    f_swin_list, cam_list, seg_list, label_list = [], [], [], []

    for idx in indices:
        filepath, orig_label = img_paths[idx]
        label = binary_map[orig_label]

        # Tensor normalizado para Swin y GradCAM
        img_pil = Image.open(filepath).convert('RGB')
        tensor = transform(img_pil).unsqueeze(0).to(device)  # (1,3,224,224)

        # Rama 1 — Swin features
        f_swin = extract_swin_features(swin, tensor)  # (1, 768)

        # Rama 2 — GradCAM++ map
        cam_map = extract_gradcam_map(cam_extractor, tensor, label)  # (1,1,224,224)

        # Rama 3 — Cellpose mask
        seg_mask = extract_cellpose_mask(cellpose_model, filepath)   # (1,1,224,224)

        f_swin_list.append(f_swin)
        cam_list.append(cam_map)
        seg_list.append(seg_mask)
        label_list.append(label)

    f_swin_batch = torch.cat(f_swin_list, dim=0)           # (B, 768)
    cam_batch    = torch.cat(cam_list,    dim=0).to(device) # (B, 1, 224, 224)
    seg_batch    = torch.cat(seg_list,    dim=0).to(device) # (B, 1, 224, 224)
    labels_batch = torch.tensor(label_list, dtype=torch.long).to(device)

    return f_swin_batch, cam_batch, seg_batch, labels_batch


# ══════════════════════════════════════════════════════════════════════════════
# MODO TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    print("\n=== MODO ENTRENAMIENTO ===\n")

    # Carga modelos base
    resnet  = load_resnet(RESNET_PTH)
    swin    = load_swin(SWIN_PTH)
    cellpose_model = load_cellpose()
    cam_extractor  = GradCAMPlusPlus(model=resnet, target_layers=[resnet.layer4[-1]])

    # Dataset base (sin transform aún — lo aplicamos en build_batch)
    raw_dataset = datasets.ImageFolder(root=TRAIN_DIR)
    binary_targets = [BINARY_MAP[label] for _, label in raw_dataset.imgs]

    train_idx, val_idx = train_test_split(
        range(len(raw_dataset)),
        test_size=0.2,
        stratify=binary_targets,
        random_state=42
    )

    # Pesos de clase para loss
    counts = Counter(binary_targets)
    weight_tensor = torch.tensor([
        (counts[0] + counts[1]) / (2 * counts[0]),
        (counts[0] + counts[1]) / (2 * counts[1])
    ]).float().to(device)
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
            f_swin, cam_map, seg_mask, labels = build_batch(
                batch_indices, raw_dataset, raw_dataset.imgs,
                swin, cam_extractor, cellpose_model,
                val_transform, BINARY_MAP   # val_transform: sin augmentation para GradCAM
            )

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
                f_swin, cam_map, seg_mask, labels = build_batch(
                    batch_indices, raw_dataset, raw_dataset.imgs,
                    swin, cam_extractor, cellpose_model,
                    val_transform, BINARY_MAP
                )
                logits = fusion(f_swin, cam_map, seg_mask)
                preds_list.append(logits.argmax(1).cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        preds_flat  = np.concatenate(preds_list)
        labels_flat = np.concatenate(labels_list)
        f1 = f1_score(labels_flat, preds_flat, pos_label=1)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss_acum/len(batches):.4f} | Val F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(fusion.state_dict(), FUSION_PTH)
            print(f"  ✓ Nuevo mejor F1: {best_f1:.4f} — guardado en {FUSION_PTH}")

    print(f"\nEntrenamiento completado. Mejor F1: {best_f1:.4f}")
    print(classification_report(labels_flat, preds_flat, target_names=['benign', 'cancer']))


# ══════════════════════════════════════════════════════════════════════════════
# MODO EVAL
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(args):
    print("\n=== MODO EVALUACIÓN ===\n")

    os.makedirs('results/correct_cancer', exist_ok=True)
    os.makedirs('results/correct_benign', exist_ok=True)
    os.makedirs('results/wrong',          exist_ok=True)

    # Carga modelos
    resnet  = load_resnet(RESNET_PTH)
    swin    = load_swin(SWIN_PTH)
    cellpose_model = load_cellpose()
    cam_extractor  = GradCAMPlusPlus(model=resnet, target_layers=[resnet.layer4[-1]])

    fusion = FusionClassifier().to(device)
    fusion.load_state_dict(torch.load(FUSION_PTH, map_location=device))
    fusion.eval()

    test_dataset = datasets.ImageFolder(root=TEST_DIR)
    all_preds, all_labels, all_probs = [], [], []

    for i, (filepath, orig_label) in enumerate(test_dataset.imgs):
        label = BINARY_MAP[orig_label]

        img_pil = Image.open(filepath).convert('RGB')
        tensor  = val_transform(img_pil).unsqueeze(0).to(device)

        # Features de las 3 ramas
        f_swin   = extract_swin_features(swin, tensor)
        cam_map  = extract_gradcam_map(cam_extractor, tensor, label).to(device)
        seg_mask = extract_cellpose_mask(cellpose_model, filepath).to(device)

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
                                  targets=[ClassifierOutputTarget(label)])[0]
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
    print("\n" + classification_report(all_labels, all_preds, target_names=['benign', 'cancer']))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print(f"AUC: {roc_auc_score(all_labels, all_probs):.3f}")
    print(f"AP:  {average_precision_score(all_labels, all_probs):.3f}")

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
                xticklabels=['benign', 'cancer'],
                yticklabels=['benign', 'cancer'])
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
import torch
from torchvision import datasets
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
from torch.utils import data
import torchvision.models as models
from torch import nn
import torch.optim as optim
from collections import Counter
import numpy as np
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import WeightedRandomSampler
from utils import device, BINARY_MAP, CitologiaDataset, TARGET_NAMES


# ─── Transforms ───────────────────────────────────────────────────────────────
# Swin espera 224×224, igual que ResNet50, sin cambios aquí

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
train_dataset = datasets.ImageFolder(root='dataset/entrenamiento', transform=train_transform)
val_dataset   = datasets.ImageFolder(root='dataset/entrenamiento', transform=val_transform)

binary_map = BINARY_MAP  # benigna=0, resto=1
train_binary_dataset = CitologiaDataset(train_dataset, binary_map)
val_binary_dataset   = CitologiaDataset(val_dataset, binary_map)

binary_targets = [binary_map[label] for label in train_dataset.targets]

train_idx, val_idx = train_test_split(
    range(len(train_binary_dataset)),
    test_size=0.2,
    stratify=binary_targets,
    random_state=42
)

train_split = data.Subset(train_binary_dataset, train_idx)
val_split   = data.Subset(val_binary_dataset,   val_idx)

# ─── Modelo: Swin-T ───────────────────────────────────────────────────────────

swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)

# La cabeza original es swin.head (Linear 768→1000)
# La sustituimos por nuestro clasificador binario (2) O CUATERNARIO (4)
swin.head = nn.Linear(768, 4)

# Estrategia de congelado:
#   - Congela patch_embed + primeras 2 stages (stages 0 y 1)
#   - Entrena stages 2 y 3 (las más semánticas) + norm final + head
#
# Arquitectura Swin-T:
#   swin.features[0]  → PatchEmbedding
#   swin.features[1]  → Stage 0  (2 bloques, C=96)
#   swin.features[2]  → PatchMerging
#   swin.features[3]  → Stage 1  (2 bloques, C=192)
#   swin.features[4]  → PatchMerging
#   swin.features[5]  → Stage 2  (6 bloques, C=384)  ← entrenamos
#   swin.features[6]  → PatchMerging
#   swin.features[7]  → Stage 3  (2 bloques, C=768)  ← entrenamos
#   swin.norm         → LayerNorm final               ← entrenamos
#   swin.head         → clasificador                  ← entrenamos

FREEZE_UP_TO = 5  # congela features[0..4], entrena features[5..7]

for i, module in enumerate(swin.features):
    requires_grad = (i >= FREEZE_UP_TO)
    for param in module.parameters():
        param.requires_grad = requires_grad

# norm y head siempre entrenables
for param in swin.norm.parameters():
    param.requires_grad = True
for param in swin.head.parameters():
    param.requires_grad = True

swin = swin.to(device)

# ─── Optimizador con LRs diferenciados ────────────────────────────────────────
# Stage 2+3: lr bajo para no destruir features preentrenadas
# Head: lr más alto para aprender la tarea nueva rápido

optimizer = optim.AdamW([
    {"params": swin.features[5].parameters(), "lr": 5e-6},
    {"params": swin.features[6].parameters(), "lr": 5e-6},
    {"params": swin.features[7].parameters(), "lr": 5e-6},
    {"params": swin.norm.parameters(),         "lr": 5e-6},
    {"params": swin.head.parameters(),         "lr": 5e-4},
], weight_decay=0.01)

# Scheduler coseno: reduce lr suavemente a lo largo del entrenamiento
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-7)

# ─── Loss con pesos de clase ───────────────────────────────────────────────────
'''
counts = Counter(binary_targets)
weight_benign    = (counts[0] + counts[1]) / (2 * counts[0])
weight_malignant = (counts[0] + counts[1]) / (2 * counts[1])
weight_tensor = torch.tensor([weight_benign, weight_malignant]).float().to(device) CLASIFICACION BINARIA
'''

counts = Counter(binary_targets)
total = sum(counts.values())
weights = [total / (4 * counts[i]) for i in range(4)]
weight_tensor = torch.tensor(weights).float().to(device)

criterion = nn.CrossEntropyLoss(weight=weight_tensor)

# ─── Entrenamiento ─────────────────────────────────────────────────────────────

EPOCHS      = 100
BATCH       = 32
NUM_WORKERS = 4  # 4 en idun

train_targets = [binary_targets[i] for i in train_idx]
sample_weights = [1.0 / counts[train_targets[i]] for i in range(len(train_idx))]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
train_loader = data.DataLoader(train_split, sampler=sampler, num_workers=NUM_WORKERS, batch_size=BATCH)
val_loader   = data.DataLoader(val_split,   shuffle=False, num_workers=NUM_WORKERS, batch_size=BATCH)

best_f1 = 0.0

for epoch in range(EPOCHS):
    # ── Train ──
    swin.train()
    loss_acum = 0.0
    for imgs, y_hat in train_loader:
        imgs, y_hat = imgs.to(device), y_hat.to(device)
        optimizer.zero_grad()
        y    = swin(imgs)
        loss = criterion(y, y_hat)
        loss.backward()
        # Gradient clipping: estabiliza el entrenamiento de transformers
        torch.nn.utils.clip_grad_norm_(swin.parameters(), max_norm=1.0)
        optimizer.step()
        loss_acum += loss.item()

    scheduler.step()
    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {loss_acum/len(train_loader):.4f} | LR head: {scheduler.get_last_lr()[-1]:.2e}")

    # ── Validation ──
    swin.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, y_hat in val_loader:
            imgs, y_hat = imgs.to(device), y_hat.to(device)
            y = swin(imgs)
            preds.append(y.argmax(1).cpu().numpy())
            labels.append(y_hat.cpu().numpy())

    preds_flat  = np.concatenate(preds)
    labels_flat = np.concatenate(labels)

   # f1 = f1_score(labels_flat, preds_flat, pos_label=1) BINARIA
    f1 = f1_score(labels_flat, preds_flat, average='macro')
    if f1 > best_f1:
        best_f1 = f1
        torch.save(swin.state_dict(), 'best_swin_4class.pth')
        print(f" Nuevo mejor F1: {best_f1:.4f} — modelo guardado")

    print(f"Epoch {epoch+1}/{EPOCHS} — Val Acc: {np.mean(preds_flat == labels_flat):.3f}")
    print(classification_report(labels_flat, preds_flat, target_names=TARGET_NAMES))
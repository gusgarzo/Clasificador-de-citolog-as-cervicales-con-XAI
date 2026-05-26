import torch
from torchvision import datasets
from torchvision.transforms import  transforms
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
val_dataset = datasets.ImageFolder(root='dataset/entrenamiento', transform=val_transform)

binary_map = BINARY_MAP  # benigna=0, resto=1
train_binary_dataset = CitologiaDataset(train_dataset, binary_map)
val_binary_dataset = CitologiaDataset(val_dataset, binary_map)

binary_targets = [binary_map[label] for label in train_dataset.targets]

train_idx, val_idx = train_test_split(
    range(len(train_binary_dataset)),
    test_size=0.2,
    stratify=binary_targets,
    random_state=42
)

train_split = data.Subset(train_binary_dataset, train_idx)
val_split = data.Subset(val_binary_dataset, val_idx)


#Model loading 
convnext = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
# convnext.classifier[2] = nn.Linear(768, 2) BINARIA
convnext.classifier[2] = nn.Linear(768, 4)


convnext = convnext.to(device)

# freezing
for i, module in enumerate(convnext.features):
    if i < 6:
        for param in module.parameters():
            param.requires_grad = False

optimizer = optim.Adam([
    {"params": convnext.features[6].parameters(), "lr": 0.000001},
    {"params": convnext.features[7].parameters(), "lr": 0.00001},
    {"params": convnext.classifier.parameters(),  "lr": 0.001}
])

counts = Counter(binary_targets)

# weight_benign = (counts[0] + counts[1])/(2*counts[0])
# weight_malignant = (counts[0] + counts[1])/(2*counts[1])
# weight_tensor = torch.tensor([weight_benign, weight_malignant]).float().to(device)  # BINARIA
total = sum(counts.values())
weights = [total / (4 * counts[i]) for i in range(4)]
weight_tensor = torch.tensor(weights).float().to(device)


criterion = nn.CrossEntropyLoss(weight=weight_tensor)

EPOCHS = 100
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
BATCH = 32
NUM_WORKERS = 4     # 4 idun

train_targets = [binary_targets[i] for i in train_idx]
sample_weights = [1.0 / counts[train_targets[i]] for i in range(len(train_idx))]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
train_loader = data.DataLoader(train_split, sampler=sampler, num_workers=NUM_WORKERS, batch_size=BATCH)

val_loader = data.DataLoader(val_split, shuffle=False, num_workers=NUM_WORKERS, batch_size= BATCH )

best_f1 = 0
for epoch in range(EPOCHS):
    # ── Train ──
    convnext.train()
    loss_acum = 0
    for imgs, y_hat in train_loader:
        imgs, y_hat = imgs.to(device), y_hat.to(device)
        optimizer.zero_grad()
        y = convnext(imgs)
        loss = criterion(y, y_hat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(convnext.parameters(), max_norm=1.0)
        optimizer.step()
        loss_acum += loss.item()

    scheduler.step()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss_acum/len(train_loader):.4f}")

    # ── Validation ──
    convnext.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, y_hat in val_loader:
            imgs, y_hat = imgs.to(device), y_hat.to(device)
            y = convnext(imgs)
            preds.append(y.argmax(1).cpu().numpy())
            labels.append(y_hat.cpu().numpy())

    preds_flat  = np.concatenate(preds)
    labels_flat = np.concatenate(labels)
    # f1 = f1_score(labels_flat, preds_flat, pos_label=1)  # BINARIA
    f1 = f1_score(labels_flat, preds_flat, average='macro')

    if f1 > best_f1:
        best_f1 = f1
        # torch.save(convnext.state_dict(), 'best_model.pth')  # BINARIA
        torch.save(convnext.state_dict(), 'best_model_4class.pth')
        print(f"  New best F1: {best_f1:.4f} — model saved")

    print(f"Epoch {epoch+1}/{EPOCHS} - Val Accuracy: {np.mean(preds_flat == labels_flat):.3f}")
    print(classification_report(labels_flat, preds_flat, target_names=TARGET_NAMES))
    



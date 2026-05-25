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


device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

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


train_dataset = datasets.ImageFolder(root='dataset/entrenamiento', transform=train_transform)
val_dataset = datasets.ImageFolder(root='dataset/entrenamiento', transform=val_transform)

binary_map = {0: 1, 1: 1, 2: 1, 3: 0}  # benigna=0, resto=1
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
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet50.fc = nn.Linear(2048, 2)

#Freezing everything except last 2 layers
for name, module in resnet50.named_children():
    if name not in ['layer4', 'fc']:
        for param in module.parameters():
            param.requires_grad = False

resnet50 =  resnet50.to(device)

optimizer = optim.Adam([{"params" : resnet50.layer4.parameters(), "lr": 0.00001},
                        {"params" : resnet50.fc.parameters(), "lr": 0.001}])

counts = Counter(binary_targets)

weight_benign = (counts[0] + counts[1])/(2*counts[0])
weight_malignant = (counts[0] + counts[1])/(2*counts[1])

weight_tensor = torch.tensor([weight_benign, weight_malignant]).float().to(device)



criterion = nn.CrossEntropyLoss(weight=weight_tensor)

EPOCHS = 20
BATCH = 32
NUM_WORKERS = 0     # 4 idun
train_loader = data.DataLoader(train_split, shuffle=True, num_workers=NUM_WORKERS, batch_size= BATCH )
val_loader = data.DataLoader(val_split, shuffle=False, num_workers=NUM_WORKERS, batch_size= BATCH )

best_f1 = 0
for epoch in range(EPOCHS):
    resnet50.train()
    loss_acum = 0
    for imgs, y_hat in train_loader:
        imgs, y_hat = imgs.to(device), y_hat.to(device)
        optimizer.zero_grad()
        y = resnet50(imgs)
        loss = criterion(y, y_hat)
        loss_acum += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss_acum/len(train_loader)}")
    #Validation
    preds = []
    labels = []
    
    resnet50.eval()
    with torch.no_grad():
        for imgs, y_hat in val_loader:
            imgs, y_hat = imgs.to(device), y_hat.to(device)
            y = resnet50(imgs)
            preds.append(y.argmax(1).cpu().numpy())
            labels.append(y_hat.cpu().numpy())

    preds_flattened = np.concatenate(preds)
    labels_flattened = np.concatenate(labels)

    f1 = f1_score(labels_flattened, preds_flattened, pos_label=1)
    if f1 > best_f1: 
        best_f1 = f1
        torch.save(resnet50.state_dict(), 'best_model.pth')

    print(f"Epoch {epoch+1}/{EPOCHS} - Val Accuracy: {np.mean(preds_flattened == labels_flattened):.3f}")
    print(classification_report(labels_flattened, preds_flattened, target_names=['benign', 'cancer']))
    



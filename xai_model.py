import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
from torch.utils import data
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import seaborn as sns

import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score


#Directories for created images
os.makedirs('results/correct_cancer', exist_ok=True)
os.makedirs('results/correct_benign', exist_ok=True)
os.makedirs('results/wrong', exist_ok=True)

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
    
resnet50 = models.resnet50(weights=None)
resnet50.fc = nn.Linear(2048, 2)
resnet50.load_state_dict(torch.load('best_model.pth', map_location=device))
resnet50 = resnet50.to(device)
resnet50.eval()

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

target_layers = [resnet50.layer4[-1]]
cam = GradCAMPlusPlus(model=resnet50, target_layers=target_layers)

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder('dataset/test', transform=val_transform)
binary_map = {0: 1, 1: 1, 2: 1, 3: 0}  
test_binary_dataset = CitologiaDataset(test_dataset, binary_map)
test_loader = data.DataLoader(test_binary_dataset, batch_size=1, shuffle=False)

all_preds = []
all_labels = []
all_probs = []

for i, (tensor, y_hat) in enumerate(test_loader):
    filepath, _ = test_dataset.imgs[i]
    original_image = Image.open(filepath).convert('RGB')
    rgb_image = np.array(original_image.resize((224, 224))) / 255.0
   
    
    tensor = tensor.to(device)
    output = resnet50(tensor)

    probs = torch.softmax(output, dim=1)
    cancer_prob = probs[0][1].item()
    all_probs.append(cancer_prob)   

    pred = output.argmax(1).item()
    y_hat = y_hat.item()
    targets = [ClassifierOutputTarget(y_hat)]

    all_preds.append(pred)
    all_labels.append(y_hat)

    grayscale_cam = cam(input_tensor=tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]
    visualization = show_cam_on_image(rgb_image.astype(np.float32), grayscale_cam, use_rgb=True)

    filename = f'image_{i}.png'
    if pred == y_hat == 1:
        save_path = f'results/correct_cancer/{filename}'
    elif pred == y_hat == 0:
        save_path = f'results/correct_benign/{filename}'
    else:
        save_path = f'results/wrong/{filename}'

    plt.imsave(save_path, visualization)

print(classification_report(all_labels, all_preds, target_names=['benign', 'cancer']))
print('Confusion matrix:')
print(confusion_matrix(all_labels, all_preds))
print(f'AUC: {roc_auc_score(all_labels, all_preds):.3f}')

#ROC curve 
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(all_labels, all_probs):.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('results/roc_curve.png')
plt.show()

#Confusion matrix 
cm = confusion_matrix(all_labels, all_preds)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['benign', 'cancer'],
            yticklabels=['benign', 'cancer'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.savefig('results/confusion_matrix.png')
plt.show()

#Precision-Recall curve
precision, recall, _ = precision_recall_curve(all_labels, all_probs)
ap = average_precision_score(all_labels, all_probs)

plt.figure()
plt.plot(recall, precision, label=f'AP = {ap:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('results/precision_recall_curve.png')
plt.show()


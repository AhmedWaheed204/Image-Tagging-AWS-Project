import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

# Define constants
TRAIN_DIR = '/kaggle/input/coco-dataset-for-multi-label-image-classification/imgs/imgs/train'
TEST_DIR = '/kaggle/input/coco-dataset-for-multi-label-image-classification/imgs/imgs/test'
LABELS_FILE = '/kaggle/input/coco-dataset-for-multi-label-image-classification/labels/labels/labels_train.csv'
CATEGORIES_FILE = '/kaggle/input/coco-dataset-for-multi-label-image-classification/labels/labels/categories.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset class (unchanged)
class MultiLabelImageDataset(Dataset):
    def __init__(self, img_dir, labels_file, categories_file, augment_classes=None, transform_aug=None, transform=None):
        self.img_dir = img_dir
        self.labels = pd.read_csv(labels_file)
        self.categories = pd.read_csv(categories_file)
        self.augment_classes = augment_classes
        self.transform_aug = transform_aug
        self.transform = transform
        self.num_classes = len(self.labels.columns) - 1  # Exclude the image name column

    def __len__(self):
        return len(self.labels) * (2 if self.augment_classes else 1)

    def __getitem__(self, idx):
        augment_idx = idx // 2 if self.augment_classes else idx
        img_name = self.labels.iloc[augment_idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        labels = torch.tensor(self.labels.iloc[augment_idx, 1:].values.astype(float), dtype=torch.float32)

        if self.augment_classes and any(labels[cls] == 1 for cls in self.augment_classes) and (idx % 2 == 1):
            if self.transform_aug:
                image = self.transform_aug(image)
        else:
            if self.transform:
                image = self.transform(image)

        return image, labels

# Define transforms
transform_regular = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Specify classes to augment (you may want to adjust this based on your specific needs)
augment_classes = range(1,80)  # Example: augment first three classes

# Create full dataset
full_dataset = MultiLabelImageDataset(TRAIN_DIR, LABELS_FILE, CATEGORIES_FILE, 
                                      augment_classes=augment_classes,
                                      transform_aug=transform_aug,
                                      transform=transform_regular)

total_size = len(full_dataset)
train_size = int(0.9 * total_size)
val_size = int(0.1 * total_size)
# test_size = total_size - (train_size + val_size)
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

num_classes = full_dataset.num_classes
print(f"Number of classes: {num_classes}")
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
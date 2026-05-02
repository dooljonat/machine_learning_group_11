# Convolutional Neural Network (CNN) model - Brady Napier

"""
Note: actual training was done through a .ipynb file in Google Colab. This .py file was created using the training .ipynb using AI so that everything would be in one place, easy to run, and structured.

Note: make sure all libraries and requirements have been properly installed and you are running in an environment where the libraries are accessible.

Usage Instructions: from the root dir (machine_learning_group_11) run: python -m models.cnn.model using the command line arguments appropriate
- To run with a saved model use: --load_model path/to/model (ex: --load_model models/cnn/results/tiny_imagenet_cnn_with_aug.pth, models available are cnn_no_aug and cnn_with_aug)
- To train a model from scratch use: --model "cnn" or --model "resnet"
    - To specify the number of epochs use: --epochs 50 (default is 30)
    - To use data augmentation use: --augment (exclude this to avoid the use of data augmentation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from dataclasses import dataclass

import copy
import time
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import os

from dataset.loader import DataConfig, get_dataloaders


# =========================================================
# DATA AUGMENTATION
# =========================================================
def add_data_augmentation(train_loader, val_loader):

    class TensorAugment:
        def __call__(self, x):

            if random.random() < 0.5:
                x = TF.hflip(x)

            angle = random.uniform(-10, 10)
            x = TF.rotate(x, angle)

            if random.random() < 0.8:
                x = TF.adjust_brightness(x, 1 + random.uniform(-0.2, 0.2))
                x = TF.adjust_contrast(x, 1 + random.uniform(-0.2, 0.2))
                x = TF.adjust_saturation(x, 1 + random.uniform(-0.2, 0.2))

            return x

    train_aug = TensorAugment()
    val_aug = lambda x: x

    class WrappedDataset(Dataset):
        def __init__(self, base_dataset, transform):
            self.base = base_dataset
            self.transform = transform

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, label = self.base[idx]
            img = self.transform(img)
            return img, label

    train_ds = WrappedDataset(train_loader.dataset, train_aug)
    val_ds = WrappedDataset(val_loader.dataset, val_aug)

    train_loader_new = DataLoader(
        train_ds,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers
    )

    val_loader_new = DataLoader(
        val_ds,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=val_loader.num_workers
    )

    return train_loader_new, val_loader_new


# =========================================================
# MODELS
# =========================================================
class TinyImageNetCNN(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)

        return out


class TinyImageNetResNet(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.layer4 = ResidualBlock(256, 512, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc(x)

        return x


# =========================================================
# TRAIN / EVAL
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_full(model, loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = 100 * correct / len(all_labels)

    # Macro precision/recall (IMPORTANT for 200 classes)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return accuracy, precision, recall, f1, all_preds, all_labels


def train(model, train_loader, val_loader, epochs=30):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_val_acc = 0
    best_model = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()

        running_loss = 0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total
        val_acc = evaluate_full(model, val_loader)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | "
              f"Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")

    model.load_state_dict(best_model)
    return model, history


# =========================================================
# PLOTTING (SAVE ONLY)
# =========================================================
def save_plots(history, model_name):

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{model_name}_loss.png")
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["val_acc"], label="Validation")
    plt.legend()
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.savefig(f"{model_name}_accuracy.png")
    plt.close()


def save_confusion_matrix_from_preds(all_preds, all_labels, model_name):

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.close()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["cnn", "resnet"], default="cnn")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--no_plots", action="store_true")

    args = parser.parse_args()

    config = DataConfig(batch_size=128, num_workers=0, max_samples=None)
    train_loader, val_loader = get_dataloaders(config)

    if args.augment:
        train_loader, val_loader = add_data_augmentation(train_loader, val_loader)

    # Model selection
    if args.model == "cnn":
        model = TinyImageNetCNN().to(device)
    else:
        model = TinyImageNetResNet().to(device)

    model_name = f"tiny_imagenet_{args.model}"

    # =========================
    # LOAD MODEL → EVAL ONLY
    # =========================
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        print("Loaded model:", args.load_model)

        acc, prec, rec, f1, preds, labels = evaluate_full(model, val_loader)

        print("\n=== Evaluation Results ===")
        print(f"Accuracy : {acc:.2f}%")
        print(f"Precision: {prec * 100:.2f}%")
        print(f"Recall   : {rec * 100:.2f}%")
        print(f"F1 Score : {f1 * 100:.2f}%")

        if not args.no_plots:
            save_confusion_matrix_from_preds(preds, labels, model_name)
            print("Confusion matrix saved!")

    # =========================
    # TRAIN → THEN EVAL
    # =========================
    else:
        model, history = train(model, train_loader, val_loader, args.epochs)

        torch.save(model.state_dict(), f"{model_name}.pth")
        print("Model saved!")

        # Final evaluation
        acc, prec, rec, f1, preds, labels = evaluate_full(model, val_loader)

        print("\n=== Final Evaluation ===")
        print(f"Accuracy : {acc:.2f}%")
        print(f"Precision: {prec:.2f}")
        print(f"Recall   : {rec:.2f}")
        print(f"F1 Score : {f1 * 100:.2f}%")

        if not args.no_plots:
            save_plots(history, model_name)
            save_confusion_matrix_from_preds(preds, labels, model_name)
            print("Plots saved!")
# Large Language Model (Decoder-only) model - Jonathan Dooley
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PATCH_SIZE = 8
IMAGE_SIZE = 64
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 64
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3         # 192
NUM_CLASSES = 200
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "llm")

CONFIGS = {
    "small": dict(embed_dim=128, num_heads=4, num_layers=4, ffn_dim=512),
    "base":  dict(embed_dim=256, num_heads=4, num_layers=6, ffn_dim=1024),
}

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PatchEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.projection = nn.Linear(PATCH_DIM, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, NUM_PATCHES, embed_dim))

    def forward(self, x):
        # x: (B, num_patches, patch_dim)
        x = self.projection(x)
        x = x + self.position_embedding
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        # split into q, k, v then reshape each into heads
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale

        # causal mask: each patch only attends to previous patches
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class FFN(nn.Module):
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ffn_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderOnlyLLM(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, ffn_dim):
        super().__init__()
        self.patch_encoder = PatchEncoder(embed_dim)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ffn_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, NUM_CLASSES)

    def forward(self, x):
        # x: (B, num_patches, patch_dim)
        x = self.patch_encoder(x)
        for block in self.decoder_blocks:
            x = block(x)
        x = self.norm(x)
        # classify using the last patch token (decoder-style)
        return self.head(x[:, -1, :])


def build_model(size="base"):
    cfg = CONFIGS[size]
    return DecoderOnlyLLM(**cfg)


def extract_patches(img_tensor):
    # img_tensor: (C, H, W) -> (num_patches, patch_dim)
    C, H, W = img_tensor.shape
    patches = img_tensor.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
    patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, C * PATCH_SIZE * PATCH_SIZE)
    return patches


class TinyImageNetDataset(Dataset):
    def __init__(self, hf_dataset, augment=False):
        self.data = hf_dataset
        self.augment = augment
        base = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
        aug = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(IMAGE_SIZE, padding=4)]
        self.transform = transforms.Compose((aug if augment else []) + base)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = self.transform(img)
        patches = extract_patches(img)
        return patches, item["label"]


def save_checkpoint(model, optimizer, epoch, val_acc, size):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"{size}_epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
    }, path)


def load_checkpoint(model, path, device):
    cp = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(cp["model_state_dict"])
    print(f"Loaded checkpoint: epoch={cp['epoch']}, val_acc={cp['val_acc']:.2f}%")
    return cp["epoch"], cp["val_acc"]


def train(size="base", epochs=40, batch_size=128, lr=3e-4):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training '{size}' model on {device}")

    print("Loading dataset...")
    ds = load_dataset("zh-plus/tiny-imagenet")
    train_loader = DataLoader(TinyImageNetDataset(ds["train"], augment=True),
                              batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(TinyImageNetDataset(ds["valid"], augment=False),
                            batch_size=batch_size, shuffle=False, num_workers=4)

    model = build_model(size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    train_accs, val_accs = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        correct_t = total_t = 0
        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(patches)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            correct_t += (logits.detach().argmax(1) == labels).sum().item()
            total_t += labels.size(0)
        train_acc = 100.0 * correct_t / total_t

        correct_v = total_v = 0
        with torch.no_grad():
            for patches, labels in val_loader:
                patches, labels = patches.to(device), labels.to(device)
                correct_v += (model(patches).argmax(1) == labels).sum().item()
                total_v += labels.size(0)
        val_acc = 100.0 * correct_v / total_v

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Epoch {epoch}/{epochs} — train_acc={train_acc:.2f}%  val_acc={val_acc:.2f}%")

        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch, val_acc, size)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, os.path.join(CHECKPOINT_DIR, f"{size}_best.pth"))

        scheduler.step()

    _plot_accuracy(train_accs, val_accs, size)
    print(f"Done. Best val acc: {best_acc:.2f}%")


def _plot_accuracy(train_accs, val_accs, size):
    plt.figure()
    epochs = range(1, len(train_accs) + 1)
    plt.plot(epochs, train_accs, label="Train")
    plt.plot(epochs, val_accs, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Decoder-Only LLM ({size}) — Tiny ImageNet")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), f"accuracy_{size}.png")
    plt.savefig(out)
    print(f"Saved accuracy plot to {out}")


if __name__ == "__main__":
    train(size="base")

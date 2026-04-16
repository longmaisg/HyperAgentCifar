import sys
import time
import math
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATA_DIR = "/Users/longmai/projects/HyperAgentCifar/data"


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_loaders():
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=False)
    return train_loader, test_loader


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item() * y.size(0)
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def main():
    device = torch.device("cpu")
    model = SmallCNN().to(device)
    n_params = count_params(model)
    assert n_params < 1_000_000, f"Too many params: {n_params}"
    print(f"PARAM_COUNT={n_params}")
    sys.stdout.flush()

    train_loader, test_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)

    # Run epoch 1 to measure time
    total_epochs = 10  # placeholder; adjusted after epoch 1
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    epoch_start = time.time()
    loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    epoch1_time = time.time() - epoch_start
    scheduler.step()

    _, val_acc = evaluate(model, test_loader, criterion, device)

    # Determine remaining epochs
    if epoch1_time > 35:
        total_epochs = 1
    else:
        remaining = max(0, math.floor(40 / epoch1_time) - 1)
        total_epochs = 1 + remaining

    # Rebuild scheduler with correct T_max (reset optimizer state is fine for lr schedule)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, last_epoch=1)

    print("EPOCH_JSON=" + json.dumps({
        "epoch": 1, "total": total_epochs,
        "loss": round(loss, 4), "acc": round(train_acc, 4),
        "epoch_sec": round(epoch1_time, 2)
    }))
    sys.stdout.flush()
    print(f"VAL_ACCURACY={val_acc:.4f}")
    sys.stdout.flush()

    for e in range(2, total_epochs + 1):
        epoch_start = time.time()
        loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        _, val_acc = evaluate(model, test_loader, criterion, device)
        print("EPOCH_JSON=" + json.dumps({
            "epoch": e, "total": total_epochs,
            "loss": round(loss, 4), "acc": round(train_acc, 4),
            "epoch_sec": round(time.time() - epoch_start, 2)
        }))
        sys.stdout.flush()
        print(f"VAL_ACCURACY={val_acc:.4f}")
        sys.stdout.flush()

    sys.exit(0)


if __name__ == "__main__":
    main()
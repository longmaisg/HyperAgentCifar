import sys
import time
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

DATA_DIR = "/Users/longmai/projects/HyperAgentCifar/data"

class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x1 = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x2 = F.max_pool2d(F.relu(self.bn2(self.conv2(x1))), 2)
        shortcut = F.max_pool2d(x1, 2)
        x2 = x2 + shortcut
        x2 = x2.flatten(1)
        x2 = F.relu(self.fc1(x2))
        x2 = self.dropout(x2)
        return self.fc2(x2)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total


def main():
    device = torch.device("cpu")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    model = SmallResNet().to(device)
    param_count = count_params(model)
    assert param_count < 1_000_000, f"Too many params: {param_count}"
    print(f"PARAM_COUNT={param_count}")
    sys.stdout.flush()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    total_epochs = 5  # placeholder, adjusted after epoch 1
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    epoch = 0
    val_acc = 0.0

    # Epoch 1
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    correct = total = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch1_time = time.time() - epoch_start
    train_loss = running_loss / total
    train_acc = correct / total

    if epoch1_time > 40:
        total_epochs = 1
    else:
        total_epochs = max(1, math.floor(45 / epoch1_time))

    # Rebuild scheduler with correct T_max
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    scheduler.step()

    val_acc = evaluate(model, test_loader, device)
    epoch = 1
    print("EPOCH_JSON=" + json.dumps({
        "epoch": epoch,
        "total": total_epochs,
        "loss": round(train_loss, 4),
        "acc": round(train_acc, 4),
        "epoch_sec": round(epoch1_time, 2)
    }))
    sys.stdout.flush()
    print(f"VAL_ACCURACY={val_acc:.4f}")
    sys.stdout.flush()

    # Remaining epochs
    for e in range(2, total_epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total
        val_acc = evaluate(model, test_loader, device)

        print("EPOCH_JSON=" + json.dumps({
            "epoch": e,
            "total": total_epochs,
            "loss": round(train_loss, 4),
            "acc": round(train_acc, 4),
            "epoch_sec": round(time.time() - epoch_start, 2)
        }))
        sys.stdout.flush()
        print(f"VAL_ACCURACY={val_acc:.4f}")
        sys.stdout.flush()

    sys.exit(0)


if __name__ == "__main__":
    main()
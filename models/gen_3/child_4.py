import sys
import time
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

DATA_DIR = "/Users/longmai/projects/HyperAgentCifar/data"

class ResNet10(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.res_branch = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.res_relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res_relu(self.res_branch(x) + x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total


def main():
    device = torch.device("cpu")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2, pin_memory=False)

    model = ResNet10().to(device)
    param_count = count_params(model)
    print(f"PARAM_COUNT={param_count}")
    sys.stdout.flush()

    assert param_count < 1_000_000, f"Parameter count {param_count} exceeds 1M limit"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # Measure first epoch to determine safe epoch count
    total_epochs = 2
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.05, epochs=total_epochs,
        steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos'
    )

    for e in range(1, total_epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        correct = total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        train_acc = correct / total

        # Adapt epoch count after first epoch
        if e == 1:
            t1 = time.time() - epoch_start
            safe_epochs = max(1, math.floor(150 / t1))
            if safe_epochs < total_epochs:
                total_epochs = safe_epochs
                # Rebuild scheduler for remaining epochs
                optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=0.05, epochs=total_epochs,
                    steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos'
                )

        val_acc = evaluate(model, test_loader, device)

        print("EPOCH_JSON=" + json.dumps({
            "epoch": e,
            "total": total_epochs,
            "loss": round(epoch_loss, 4),
            "acc": round(train_acc, 4),
            "epoch_sec": round(time.time() - epoch_start, 2),
        }))
        sys.stdout.flush()
        print(f"VAL_ACCURACY={val_acc:.4f}")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
import sys
import time
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

DATA_DIR = "/Users/longmai/projects/HyperAgentCifar/data"

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_loaders():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2, pin_memory=False)
    return train_loader, test_loader


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


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = correct = total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return running_loss / total, correct / total


def main():
    device = torch.device("cpu")
    model = SmallCNN().to(device)

    param_count = count_params(model)
    print(f"PARAM_COUNT={param_count}")
    sys.stdout.flush()

    assert param_count < 1_000_000, f"Model has {param_count} params, must be under 1,000,000"

    train_loader, test_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)

    # Determine epochs dynamically after first epoch
    total_epochs = 1  # will be updated after epoch 1

    scheduler = None  # will init after we know total_epochs

    val_acc = 0.0
    for e in range(1, 100):  # upper bound; break logic handles actual stopping
        epoch_start = time.time()
        loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        epoch_time = time.time() - epoch_start

        if e == 1:
            remaining = max(1, math.floor(45 / epoch_time))
            total_epochs = 1 + remaining
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

        if scheduler is not None:
            scheduler.step()

        val_acc = evaluate(model, test_loader, device)

        print("EPOCH_JSON=" + json.dumps({
            "epoch": e,
            "total": total_epochs,
            "loss": round(loss, 4),
            "acc": round(train_acc, 4),
            "epoch_sec": round(epoch_time, 2),
        }))
        sys.stdout.flush()
        print(f"VAL_ACCURACY={val_acc:.4f}")
        sys.stdout.flush()

        if e >= total_epochs:
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
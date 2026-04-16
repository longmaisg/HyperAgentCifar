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

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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

    model = SmallCNN().to(device)
    param_count = count_params(model)
    assert param_count < 1_000_000, f"Model has {param_count} params, must be < 1M"
    print(f"PARAM_COUNT={param_count}")
    sys.stdout.flush()

    criterion = nn.CrossEntropyLoss()
    total_epochs = 10  # placeholder, adjusted after epoch 1

    optimizer = optim.SGD(model.parameters(), lr=0.10, momentum=0.9, weight_decay=1e-4)
    scheduler = None  # will reinit after epoch 1

    val_acc = 0.0
    epoch1_time = None

    e = 0
    while True:
        e += 1
        epoch_start = time.time()

        model.train()
        running_loss = 0.0
        correct = total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / total
        train_acc = correct / total

        if e == 1:
            epoch1_time = epoch_time
            if epoch1_time > 40:
                total_epochs = 1
            else:
                total_epochs = max(1, math.floor(45 / epoch1_time))
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

        if scheduler is not None:
            scheduler.step()

        val_acc = evaluate(model, test_loader, device)

        print("EPOCH_JSON=" + json.dumps({
            "epoch": e,
            "total": total_epochs,
            "loss": round(avg_loss, 4),
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
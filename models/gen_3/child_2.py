import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

DATA_DIR = "/Users/longmai/projects/HyperAgentCifar/data"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model, loader, criterion, device):
    model.eval()
    correct = total = loss_sum = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss_sum += criterion(outputs, targets).item() * targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return loss_sum / total, correct / total

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

    train_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=False)

    model = Net().to(device)
    param_count = count_params(model)
    assert param_count < 1_000_000, f"Model has {param_count} params, exceeds 1M limit"
    print(f"PARAM_COUNT={param_count}")
    sys.stdout.flush()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    total_epochs = 2
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.1, epochs=total_epochs,
        steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos'
    )

    for e in range(1, total_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_sum = train_correct = train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets).sum().item()
            train_total += targets.size(0)

        loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        _, val_acc = evaluate(model, test_loader, criterion, device)

        print("EPOCH_JSON=" + json.dumps({
            "epoch": e, "total": total_epochs,
            "loss": round(loss, 4), "acc": round(train_acc, 4),
            "epoch_sec": round(time.time() - epoch_start, 2)
        }))
        sys.stdout.flush()
        print(f"VAL_ACCURACY={val_acc:.4f}")
        sys.stdout.flush()

    return 0

if __name__ == "__main__":
    sys.exit(main())
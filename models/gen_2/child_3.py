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
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64), nn.ReLU(),
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
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)

    batch_size = 512
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0)

    model = SmallCNN().to(device)
    param_count = count_params(model)
    print(f"PARAM_COUNT={param_count}")
    sys.stdout.flush()

    assert param_count < 1_000_000, f"Too many parameters: {param_count}"

    optimizer = optim.SGD(model.parameters(), lr=0.10, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # We'll set total_epochs after measuring epoch 1
    total_epochs = 10  # placeholder; adjusted dynamically
    scheduler = None  # created after epoch 1

    val_acc = 0.0
    epoch1_time = None

    e = 0
    while True:
        e += 1
        if e > total_epochs:
            break

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

            running_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_time = time.time() - epoch_start

        avg_loss = running_loss / total
        train_acc = correct / total

        val_acc = evaluate(model, test_loader, device)

        if scheduler is not None:
            scheduler.step()

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

        if e == 1:
            epoch1_time = epoch_time
            if epoch1_time > 45:
                break
            remaining = max(1, math.floor(50 / epoch1_time))
            total_epochs = 1 + remaining
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
            # Step scheduler once for the epoch we already completed
            scheduler.step()

    sys.exit(0)


if __name__ == "__main__":
    main()
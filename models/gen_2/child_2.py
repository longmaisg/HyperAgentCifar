import sys
import time
import json
import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

DATA_DIR = "/Users/longmai/projects/HyperAgentCifar/data"

class FastNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 8 * 8, 96), nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(96, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    device = torch.device("cpu")

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=0, pin_memory=False)

    model = FastNet().to(device)
    n_params = count_params(model)
    assert n_params < 1_000_000, f"Too many params: {n_params}"
    print(f"PARAM_COUNT={n_params}")
    sys.stdout.flush()

    criterion = nn.CrossEntropyLoss()
    total_epochs = 10  # placeholder; adjusted after epoch 1

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    epoch = 0
    while True:
        epoch += 1
        epoch_start = time.time()

        model.train()
        running_loss = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        epoch_time = time.time() - epoch_start
        train_loss = running_loss / total
        train_acc  = correct / total

        # Adjust total_epochs after first epoch
        if epoch == 1:
            if epoch_time > 40:
                total_epochs = 1
            else:
                total_epochs = max(1, math.floor(45 / epoch_time))
            # Recreate scheduler with correct T_max
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

        scheduler.step()

        val_acc = evaluate(model, test_loader, device)

        print("EPOCH_JSON=" + json.dumps({
            "epoch": epoch,
            "total": total_epochs,
            "loss": round(train_loss, 4),
            "acc": round(train_acc, 4),
            "epoch_sec": round(epoch_time, 2),
        }))
        sys.stdout.flush()
        print(f"VAL_ACCURACY={val_acc:.4f}")
        sys.stdout.flush()

        if epoch >= total_epochs:
            break

    sys.exit(0)

if __name__ == "__main__":
    main()
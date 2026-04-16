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


class DSBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.pw(self.dw(x)))))


class TinyDSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = DSBlock(3, 32)
        self.block2 = DSBlock(32, 64)
        self.block3 = DSBlock(64, 64)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_loaders():
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=False)
    return train_loader, test_loader


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


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def main():
    device = torch.device("cpu")

    model = TinyDSNet().to(device)
    n_params = count_params(model)
    assert n_params < 1_000_000, f"Too many params: {n_params}"
    print(f"PARAM_COUNT={n_params}")
    sys.stdout.flush()

    train_loader, test_loader = get_loaders()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    total_epochs = 5  # placeholder; adjusted after epoch 1

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = None  # created after we know total_epochs

    val_acc = 0.0
    e = 0
    while True:
        e += 1
        epoch_start = time.time()
        loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        epoch_sec = time.time() - epoch_start

        if e == 1:
            total_epochs = max(2, int(50 / epoch_sec))
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

        scheduler.step()
        val_acc = evaluate(model, test_loader, device)

        print("EPOCH_JSON=" + json.dumps({
            "epoch": e,
            "total": total_epochs,
            "loss": round(loss, 4),
            "acc": round(train_acc, 4),
            "epoch_sec": round(epoch_sec, 2),
        }))
        sys.stdout.flush()
        print(f"VAL_ACCURACY={val_acc:.4f}")
        sys.stdout.flush()

        if e >= total_epochs:
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
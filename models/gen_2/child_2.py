import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATA_DIR = "/Users/longmai/projects/HyperAgentCifar/data"

class WiderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(96, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(192 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = WiderCNN().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"PARAM_COUNT={param_count}")
    assert param_count < 1_000_000, f"Too many params: {param_count}"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(10):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs).argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    val_accuracy = correct / total
    print(f"VAL_ACCURACY={val_accuracy:.4f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
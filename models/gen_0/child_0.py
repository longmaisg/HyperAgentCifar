import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATA_DIR = "/Users/longmai/projects/HyperAgentCifar/data"

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    train_dataset = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    model = SimpleCNN().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"PARAM_COUNT={param_count}")
    sys.stdout.flush()

    assert param_count < 1_000_000, f"Model has {param_count} parameters, exceeds 1M limit"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    total_epochs = 10
    for epoch in range(1, total_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += inputs.size(0)
        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print("EPOCH_JSON=" + json.dumps({
            "epoch": epoch, "total": total_epochs,
            "loss": round(epoch_loss, 4), "acc": round(epoch_acc, 4)
        }))
        sys.stdout.flush()

    model.eval()
    correct = 0
    total = 0
    total_batches = len(test_loader)
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += inputs.size(0)
            if batch_idx % 10 == 0:
                print(f"Test batch {batch_idx}/{total_batches}")
                sys.stdout.flush()

    val_accuracy = correct / total
    print(f"VAL_ACCURACY={val_accuracy}")
    sys.stdout.flush()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
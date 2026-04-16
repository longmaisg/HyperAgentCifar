import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
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

def main():
    try:
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
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

        device = torch.device("cpu")
        model = Net().to(device)

        param_count = sum(p.numel() for p in model.parameters())
        print(f"PARAM_COUNT={param_count}")
        assert param_count < 1_000_000, f"Model has {param_count} params, exceeds 1M limit"
        sys.stdout.flush()

        optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
        criterion = nn.CrossEntropyLoss()

        total_epochs = 3

        for epoch in range(1, total_epochs + 1):
            epoch_start = time.time()
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

            train_loss = running_loss / total
            train_acc = correct / total

            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += inputs.size(0)

            val_acc = val_correct / val_total
            scheduler.step()

            print("EPOCH_JSON=" + json.dumps({
                "epoch": epoch,
                "total": total_epochs,
                "loss": round(train_loss, 4),
                "acc": round(train_acc, 4),
                "epoch_sec": round(time.time() - epoch_start, 2)
            }))
            sys.stdout.flush()
            print(f"VAL_ACCURACY={val_acc:.4f}")
            sys.stdout.flush()

        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
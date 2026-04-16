# Mutation — Narrower+Deeper, Smaller FC (architecture)

**Rationale:** Replace the wide final conv block and large FC with a narrower channel progression and smaller head. Reduces params from ~621K to ~190K, cuts per-epoch time, and leaves room to safely train 3 full epochs within budget.

## Architecture
- Input: 32x32 RGB (CIFAR-10, 10 classes)
- Conv block 1: Conv2d(3→32, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 2: Conv2d(32→64, 3x3, pad=1) → BN → ReLU → MaxPool(2)
- Conv block 3: Conv2d(64→64, 3x3, pad=1) → BN → ReLU → MaxPool(2)  ← **changed** (128→64)
- Flatten → Linear(1024→128) → ReLU → Dropout(0.4) → Linear(128→10)  ← **changed**
- Est. params: ~189K

## Training
- Optimizer: SGD, lr=0.01, momentum=0.9, weight_decay=1e-4
- Scheduler: StepLR(step_size=5, gamma=0.5)
- Epochs: 3
- Batch size: 128
- Loss: CrossEntropyLoss

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616)

---
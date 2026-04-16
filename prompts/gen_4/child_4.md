# Explore — Depthwise Separable Convolutions + Global Average Pooling + Cosine Annealing

## Rationale
All prior children use standard convolutions with a large flatten→FC bottleneck. Depthwise separable convolutions (MobileNet-style) reduce multiply-adds by ~8–9× per spatial block at the cost of some representational capacity — a trade-off that is net-positive when the time budget is the binding constraint. Replacing the 2048→256 FC with Global Average Pooling eliminates the largest single weight matrix. Cosine annealing without restarts (CosineAnnealingLR) replaces StepLR; its smooth decay tends to outperform step schedules on short runs. The wider channel counts (64→128→256) compensate for the expressiveness loss from depthwise factorisation. This design targets ~150 K params and ~50–60 s total training time.

## Architecture
- Input: 32×32 RGB (CIFAR-10, 10 classes)
- Stem: Conv2d(3→64, 3×3, pad=1) → BN → ReLU → MaxPool(2)  — output: 16×16×64
- DSBlock 1: DepthwiseConv2d(64, 3×3, pad=1) → BN → ReLU → PointwiseConv2d(64→128, 1×1) → BN → ReLU → MaxPool(2)  — output: 8×8×128
- DSBlock 2: DepthwiseConv2d(128, 3×3, pad=1) → BN → ReLU → PointwiseConv2d(128→256, 1×1) → BN → ReLU → MaxPool(2)  — output: 4×4×256
- DSBlock 3: DepthwiseConv2d(256, 3×3, pad=1) → BN → ReLU → PointwiseConv2d(256→256, 1×1) → BN → ReLU  — output: 4×4×256
- Global Average Pooling → 256-dim vector
- Dropout(0.3) → Linear(256→10)

## Parameter count (estimated)
~116 K — far under limit, very fast forward/backward pass.

## Implementation note for depthwise separable block
```python
class DSBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return F.relu(self.bn2(self.pw(F.relu(self.bn1(self.dw(x))))))
```

## Training
- Optimizer: SGD, lr=0.05, momentum=0.9, weight_decay=1e-4
- Scheduler: CosineAnnealingLR(T_max=2, eta_min=1e-4)
- Epochs: 2
- Batch size: 256
- Loss: CrossEntropyLoss(label_smoothing=0.1)
- **Mixed precision: `torch.cuda.amp.autocast()` + `GradScaler`**

## Augmentation
- RandomHorizontalFlip
- RandomCrop(32, padding=4)
- Normalize mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
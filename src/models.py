"""
Network definitions.

Model A - Standard CNN    : 4 x [Conv3x3 -> BN -> ReLU -> MaxPool2x2]
Model B - Lightweight CNN : 4 x [DWConv3x3 -> BN -> ReLU -> PWConv1x1 -> BN -> ReLU -> MaxPool2x2]

Both models share the same channel progression (32-64-128-256) and the same
classifier head (GlobalAvgPool -> FC 64 -> ReLU -> Dropout -> FC 10), so the
ONLY difference between them is the type of convolution. Any change in
parameters / MACs / accuracy can therefore be attributed to the depthwise
separable factorisation.

SOTA models : MobileNetV2 and ShuffleNetV2 (x1.0), ImageNet pre-trained, with the
              final classifier replaced by a 10-way linear layer.

Hand-calculated parameter counts are in docs/architecture.md and are checked
against PyTorch by model_summary.py.
"""
import torch.nn as nn
from torchvision import models as tvm

CHANNELS = (32, 64, 128, 256)
FC_UNITS = 64
DROPOUT = 0.3


class ConvBNReLU(nn.Sequential):
    """Standard 3x3 convolution. Bias omitted because BatchNorm adds its own shift."""

    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class DSConvBNReLU(nn.Sequential):
    """Depthwise separable convolution (MobileNetV1 style).

    depthwise : one 3x3 filter per input channel (groups=cin)  -> 9*cin weights
    pointwise : 1x1 convolution mixing channels                -> cin*cout weights
    versus a standard 3x3 conv with 9*cin*cout weights.
    """

    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cin, kernel_size=3, padding=1, groups=cin, bias=False),
            nn.BatchNorm2d(cin),
            nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, kernel_size=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class CustomCNN(nn.Module):
    def __init__(self, block, num_classes=10, channels=CHANNELS, fc_units=FC_UNITS, dropout=DROPOUT):
        super().__init__()
        layers, cin = [], 3
        for cout in channels:
            # 64x64 -> 32 -> 16 -> 8 -> 4 spatial resolution
            layers += [block(cin, cout), nn.MaxPool2d(2)]
            cin = cout
        self.features = nn.Sequential(*layers)
        # Global average pooling instead of Flatten keeps the FC layer tiny
        # (256 inputs instead of 256*4*4 = 4096) - essential for the 100k budget.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cin, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_units, num_classes),  # logits; softmax only inside the loss
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def model_a(num_classes=10):
    return CustomCNN(ConvBNReLU, num_classes)


def model_b(num_classes=10):
    return CustomCNN(DSConvBNReLU, num_classes)


def mobilenet_v2(num_classes=10, pretrained=True):
    weights = tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    m = tvm.mobilenet_v2(weights=weights)
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m


def shufflenet_v2(num_classes=10, pretrained=True):
    weights = tvm.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
    m = tvm.shufflenet_v2_x1_0(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def head_parameters(name, model):
    """Parameters of the newly added classifier (trained with a higher LR)."""
    return model.classifier.parameters() if name == "mobilenet_v2" else model.fc.parameters()


BUILDERS = {
    "model_a": model_a,
    "model_b": model_b,
    "mobilenet_v2": mobilenet_v2,
    "shufflenet_v2": shufflenet_v2,
}

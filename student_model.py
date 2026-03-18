import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class StudentResNet(nn.Module):
    """
    A larger student CNN intentionally sized around ~200MB (FP32 parameters)
    to retain enough capacity for strong knowledge distillation.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        layers: tuple[int, int, int, int] = (3, 4, 6, 3),
        channels: tuple[int, int, int, int] = (156, 312, 468, 624),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = channels[0]

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(channels[0], layers[0], stride=1)
        self.layer2 = self._make_layer(channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(channels[3], layers[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[3], num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        blocks = [BasicBlock(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            blocks.append(BasicBlock(self.in_channels, out_channels, stride=1))

        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


def build_student_model_200mb(num_classes: int = 1000) -> StudentResNet:
    return StudentResNet(num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_model_size_mb(model: nn.Module) -> float:
    # FP32 parameters: 4 bytes per parameter.
    return (count_parameters(model) * 4) / (1024 * 1024)


if __name__ == "__main__":
    model = build_student_model_200mb(num_classes=1000)
    total_params = count_parameters(model)
    size_mb = estimate_model_size_mb(model)

    print(f"Total parameters: {total_params:,}")
    print(f"Approx model size (FP32 params only): {size_mb:.2f} MB")

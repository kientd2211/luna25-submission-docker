import torch
import torch.nn as nn
import torchvision.models.video as video_models

class ResNet3D(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super().__init__()

        self.resnet3d = video_models.r3d_18(pretrained=pretrained)

        # 🔴 THAY Conv3d đầu tiên: 3 → 1 channel
        old_conv = self.resnet3d.stem[0]
        self.resnet3d.stem[0] = nn.Conv3d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # Thay FC
        self.resnet3d.fc = nn.Linear(
            self.resnet3d.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.resnet3d(x)

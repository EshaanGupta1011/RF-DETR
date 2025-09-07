import torch
import torch.nn as nn

"""
Originally, DETR uses ResNET-50/101 trained on ImageNet dataset as the backbone. 
However, for simplicity and efficiency, we implement a smaller custom CNN backbone here.
"""

class DeeperCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.SiLU()
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2),
            nn.BatchNorm2d(64)
        )

        self.residual_block = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.SiLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.SiLU()
        )

        self.downsample_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2),
            nn.BatchNorm2d(256)
        )

        self. residual_block_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.SiLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.SiLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm2d(256)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )


    def forward(self, x):
        x = self.block1(x)
        res = self.downsample(x)
        x = self.residual_block(x)
        x = x + res
        x = self.block2(x)
        res = self.downsample_2(x)
        x = self.residual_block_2(x)
        x = x + res
        x = self.block3(x)
        features = self.proj(x)
        return features
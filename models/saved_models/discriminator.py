# models/discriminator.py
import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, batch_norm=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # VGG-style discriminator / Patch-based
        sequence = []
        sequence += [nn.Conv2d(in_channels, 64, 3, 1, 1), nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(256, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(512, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True)]

        self.features = nn.Sequential(*sequence)

        # classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

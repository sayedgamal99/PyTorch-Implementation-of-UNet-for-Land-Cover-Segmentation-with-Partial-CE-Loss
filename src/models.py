import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, base_channels=64, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        self.up1 = Up(base_channels * 16, base_channels *
                      8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class NestedUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x_deep, x_skip):
        x_deep = self.up(x_deep)

        diffY = x_skip.size()[2] - x_deep.size()[2]
        diffX = x_skip.size()[3] - x_deep.size()[3]

        x_deep = F.pad(x_deep, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])

        x = torch.cat([x_skip, x_deep], dim=1)
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, base_channels=64):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.conv0_0 = DoubleConv(in_channels, base_channels)
        self.conv1_0 = DoubleConv(base_channels, base_channels * 2)
        self.conv2_0 = DoubleConv(base_channels * 2, base_channels * 4)
        self.conv3_0 = DoubleConv(base_channels * 4, base_channels * 8)
        self.conv4_0 = DoubleConv(base_channels * 8, base_channels * 16)

        self.up1_0 = NestedUp(base_channels * 2, base_channels)
        self.up2_0 = NestedUp(base_channels * 4, base_channels * 2)
        self.up2_1 = NestedUp(base_channels * 2, base_channels)
        self.up3_0 = NestedUp(base_channels * 8, base_channels * 4)
        self.up3_1 = NestedUp(base_channels * 4, base_channels * 2)
        self.up3_2 = NestedUp(base_channels * 2, base_channels)
        self.up4_0 = NestedUp(base_channels * 16, base_channels * 8)
        self.up4_1 = NestedUp(base_channels * 8, base_channels * 4)
        self.up4_2 = NestedUp(base_channels * 4, base_channels * 2)
        self.up4_3 = NestedUp(base_channels * 2, base_channels)

        self.final = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.up1_0(x1_0, x0_0)

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.up2_0(x2_0, x1_0)
        x0_2 = self.up2_1(x1_1, x0_1)

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.up3_0(x3_0, x2_0)
        x1_2 = self.up3_1(x2_1, x1_1)
        x0_3 = self.up3_2(x1_2, x0_2)

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.up4_0(x4_0, x3_0)
        x2_2 = self.up4_1(x3_1, x2_1)
        x1_3 = self.up4_2(x2_2, x1_2)
        x0_4 = self.up4_3(x1_3, x0_3)

        output = self.final(x0_4)
        return output


def get_unet(model_type="unet", classes=5, in_channels=3):
    if model_type == "unet":
        model = UNet(in_channels=in_channels,
                     num_classes=classes, base_channels=64)
        print(f"Created local UNet implementation")

    elif model_type == "unetplusplus":
        model = UNetPlusPlus(in_channels=in_channels,
                             num_classes=classes, base_channels=64)
        print(f"Created local UNet++ implementation")

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Supported types: 'unet', 'unetplusplus'")

    return model

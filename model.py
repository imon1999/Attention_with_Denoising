import torch
import torch.nn as nn
import torch.nn.functional as F
from partialconv2d import PartialConv2d

# Channel Attention Module (CBAM Component)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# Spatial Attention Module (CBAM Component)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# CBAM: Combining Channel and Spatial Attention
class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# Encoder Block (Using PartialConv2D)
class EncodeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, flag):
        super(EncodeBlock, self).__init__()
        self.conv = PartialConv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.nonlinear = nn.LeakyReLU(0.1)
        self.MaxPool = nn.MaxPool2d(2)
        self.flag = flag

    def forward(self, x, mask_in):
        out1, mask_out = self.conv(x, mask_in=mask_in)
        out2 = self.nonlinear(out1)
        if self.flag:
            out = self.MaxPool(out2)
            mask_out = self.MaxPool(mask_out)
        else:
            out = out2
        return out, mask_out

# Decoder Block (Standard Convolution)
class DecodeBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, final_channel=3, p=0.7, flag=False):
        super(DecodeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channel, final_channel, kernel_size=3, padding=1)
        self.nonlinear1 = nn.LeakyReLU(0.1)
        self.nonlinear2 = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
        self.flag = flag
        self.Dropout = nn.Dropout(p)

    def forward(self, x):
        out1 = self.conv1(self.Dropout(x))
        out2 = self.nonlinear1(out1)
        out3 = self.conv2(self.Dropout(out2))
        out4 = self.nonlinear2(out3)
        if self.flag:
            out5 = self.conv3(self.Dropout(out4))
            out = self.sigmoid(out5)
        else:
            out = out4
        return out

# Main Model: DeepSelf2Self with additional encoder-decoder layers and CBAM
class DeepSelf2Self(nn.Module):
    def __init__(self, in_channel):
        super(DeepSelf2Self, self).__init__()
        # Encoder blocks
        self.EB0 = EncodeBlock(in_channel, 48, flag=False)
        self.EB1 = EncodeBlock(48, 48, flag=True)
        self.EB2 = EncodeBlock(48, 48, flag=True)
        self.EB3 = EncodeBlock(48, 48, flag=True)
        self.EB4 = EncodeBlock(48, 48, flag=True)
        self.EB5 = EncodeBlock(48, 48, flag=True)
        self.EB6 = EncodeBlock(48, 48, flag=False)
        self.EB7 = EncodeBlock(48, 96, flag=True)
        self.EB8 = EncodeBlock(96, 96, flag=True)

        self.cbam = CBAM(in_planes=96)

        # Decoder blocks
        self.DB1 = DecodeBlock(96 + 96, 96, 96)
        self.DB2 = DecodeBlock(96 + 48, 96, 96)
        self.DB3 = DecodeBlock(96 + 48, 96, 96)
        self.DB4 = DecodeBlock(96 + 48, 96, 96)
        self.DB5 = DecodeBlock(96 + 48, 96, 96)
        self.DB6 = DecodeBlock(96 + 48, 96, 96)
        self.DB7 = DecodeBlock(96 + 48, 96, 96)
        self.DB8 = DecodeBlock(96 + in_channel, 64, 32, in_channel, flag=True)

        self.concat_dim = 1

    def forward(self, x, mask):
        # Encoding Path
        out_EB0, mask = self.EB0(x, mask)
        out_EB1, mask = self.EB1(out_EB0, mask)
        out_EB2, mask = self.EB2(out_EB1, mask)
        out_EB3, mask = self.EB3(out_EB2, mask)
        out_EB4, mask = self.EB4(out_EB3, mask)
        out_EB5, mask = self.EB5(out_EB4, mask)
        out_EB6, mask = self.EB6(out_EB5, mask)
        out_EB7, mask = self.EB7(out_EB6, mask)
        out_EB8, mask = self.EB8(out_EB7, mask)

        out_EB8 = self.cbam(out_EB8)

        # Decoding Path
        out_EB8_up = F.interpolate(out_EB8, scale_factor=2, mode='bilinear', align_corners=False)
        in_DB1 = torch.cat((out_EB8_up, out_EB7), dim=1)
        out_DB1 = self.DB1(in_DB1)

        out_DB1_up = F.interpolate(out_DB1, scale_factor=2, mode='bilinear', align_corners=False)
        in_DB2 = torch.cat((out_DB1_up, out_EB6), dim=1)
        out_DB2 = self.DB2(in_DB2)

        out_DB2_up = F.interpolate(out_DB2, scale_factor=2, mode='bilinear', align_corners=False)
        out_EB5_up = F.interpolate(out_EB5, size=out_DB2_up.shape[2:], mode='bilinear', align_corners=False)
        in_DB3 = torch.cat((out_DB2_up, out_EB5_up), dim=1)
        out_DB3 = self.DB3(in_DB3)

        out_DB3_up = F.interpolate(out_DB3, scale_factor=2, mode='bilinear', align_corners=False)
        out_EB4_up = F.interpolate(out_EB4, size=out_DB3_up.shape[2:], mode='bilinear', align_corners=False)
        in_DB4 = torch.cat((out_DB3_up, out_EB4_up), dim=1)
        out_DB4 = self.DB4(in_DB4)

        out_DB4_up = F.interpolate(out_DB4, scale_factor=2, mode='bilinear', align_corners=False)
        out_EB3_up = F.interpolate(out_EB3, size=out_DB4_up.shape[2:], mode='bilinear', align_corners=False)
        in_DB5 = torch.cat((out_DB4_up, out_EB3_up), dim=1)
        out_DB5 = self.DB5(in_DB5)

        out_DB5_up = F.interpolate(out_DB5, scale_factor=2, mode='bilinear', align_corners=False)
        out_EB2_up = F.interpolate(out_EB2, size=out_DB5_up.shape[2:], mode='bilinear', align_corners=False)
        in_DB6 = torch.cat((out_DB5_up, out_EB2_up), dim=1)
        out_DB6 = self.DB6(in_DB6)

        out_DB6_up = F.interpolate(out_DB6, scale_factor=2, mode='bilinear', align_corners=False)
        out_EB1_up = F.interpolate(out_EB1, size=out_DB6_up.shape[2:], mode='bilinear', align_corners=False)
        in_DB7 = torch.cat((out_DB6_up, out_EB1_up), dim=1)
        out_DB7 = self.DB7(in_DB7)

        out_DB7_up = F.interpolate(out_DB7, scale_factor=2, mode='bilinear', align_corners=False)
        x_up = F.interpolate(x, size=out_DB7_up.shape[2:], mode='bilinear', align_corners=False)
        in_DB8 = torch.cat((out_DB7_up, x_up), dim=1)
        out_DB8 = self.DB8(in_DB8)

        final_output = F.interpolate(out_DB8, size=x.shape[2:], mode='bilinear', align_corners=False)
        return final_output

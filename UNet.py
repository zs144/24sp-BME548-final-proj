# This file contains s self-implemented U-Net model

import torch
import torch.nn.functional as F
import torchvision

class UNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.physical = self.physical_layer(in_channels)
        self.encode1  = self.encoder_block(1, 32, 7, 3)
        self.encode2  = self.encoder_block(32, 64, 3, 1)
        self.encode3  = self.encoder_block(64, 128, 3, 1)
        self.bridge   = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1,
                                     output_padding=1),
            torch.nn.AvgPool2d(2, stride=2)
        )
        self.decode3 = self.decoder_block(128*2, 64, 3, 1)
        self.decode2 = self.decoder_block(64*2, 32, 3, 1)
        self.decode1 = self.decoder_block(32*2, out_channels, 3, 1)

    def __call__(self, x):
        x = self.physical(x)

        conv1 = self.encode1(x)
        conv2 = self.encode2(conv1)
        conv3 = self.encode3(conv2)

        mid = self.bridge(conv3)

        upconv3 = self.decode3(torch.cat([mid, conv3], 1))
        upconv2 = self.decode2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.decode1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def physical_layer(self, in_channels):
        return torch.nn.Conv2d(in_channels, out_channels=1, kernel_size=1)

    def encoder_block(self, in_channels, out_channels, kernel_size, padding):
        encoding = torch.nn.Sequential(
            # CONV -> BatchNorm -> ReLU (not change the w/h of the images)
            # in_channels < out_channels
            torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                            stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            # CONV -> BatchNorm -> ReLU (not change the w/h of the images)
            torch.nn.Conv2d(out_channels, out_channels, kernel_size,
                            stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            # MaxPool layer: halve the size of the image
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        return encoding

    def decoder_block(self, in_channels, out_channels, kernel_size, padding):
        decoding = torch.nn.Sequential(
            # CONV -> BatchNorm -> ReLU (not change the w/h of the images)
            # in_channels > out_channels
            torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                            stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            # CONV -> BatchNorm -> ReLU (not change the w/h of the images)
            torch.nn.Conv2d(out_channels, out_channels, kernel_size,
                            stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            # Upsampling by doubling the size of the image
            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3,
                                     stride=2, padding=1, output_padding=1)
        )

        return decoding


class UNet_no_physical(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encode1  = self.encoder_block(4, 32, 7, 3)
        self.encode2  = self.encoder_block(32, 64, 3, 1)
        self.encode3  = self.encoder_block(64, 128, 3, 1)
        self.bridge   = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1,
                                     output_padding=1),
            torch.nn.AvgPool2d(2, stride=2)
        )
        self.decode3 = self.decoder_block(128*2, 64, 3, 1)
        self.decode2 = self.decoder_block(64*2, 32, 3, 1)
        self.decode1 = self.decoder_block(32*2, out_channels, 3, 1)

    def __call__(self, x):
        conv1 = self.encode1(x)
        conv2 = self.encode2(conv1)
        conv3 = self.encode3(conv2)

        mid = self.bridge(conv3)

        upconv3 = self.decode3(torch.cat([mid, conv3], 1))
        upconv2 = self.decode2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.decode1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def encoder_block(self, in_channels, out_channels, kernel_size, padding):
        encoding = torch.nn.Sequential(
            # CONV -> BatchNorm -> ReLU (not change the w/h of the images)
            # in_channels < out_channels
            torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                            stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            # CONV -> BatchNorm -> ReLU (not change the w/h of the images)
            torch.nn.Conv2d(out_channels, out_channels, kernel_size,
                            stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            # MaxPool layer: halve the size of the image
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        return encoding

    def decoder_block(self, in_channels, out_channels, kernel_size, padding):
        decoding = torch.nn.Sequential(
            # CONV -> BatchNorm -> ReLU (not change the w/h of the images)
            # in_channels > out_channels
            torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                            stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            # CONV -> BatchNorm -> ReLU (not change the w/h of the images)
            torch.nn.Conv2d(out_channels, out_channels, kernel_size,
                            stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            # Upsampling by doubling the size of the image
            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3,
                                     stride=2, padding=1, output_padding=1)
        )

        return decoding
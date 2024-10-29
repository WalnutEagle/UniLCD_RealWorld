import logging
import torch
import torch.nn as nn
import timm

class CustomRegNetY00(nn.Module):
    def __init__(self):
        super(CustomRegNetY00, self).__init__()
        # Create the base RegNetY_002 model
        self.model = timm.create_model('regnety_002', pretrained=True)

        # Modify the first convolution layer to accept 4 channels (RGB + Depth)
        self.model.stem.conv = nn.Conv2d(
            1,  # Change input channels from 3 (RGB) to 4 (RGB + Depth)
            self.model.stem.conv.out_channels, 
            kernel_size=self.model.stem.conv.kernel_size, 
            stride=self.model.stem.conv.stride, 
            padding=self.model.stem.conv.padding,
            bias=False
        )

        # Define the final layers for output (steering and throttle)
        self.lin = nn.Sequential(
            nn.Linear(self.model.head.fc.in_features, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 2)  # 2 outputs: steering and throttle
        )

    def forward(self, x):
        # Forward pass through the model
        x = self.model.s1(x)
        x = self.model.s2(x)
        x = self.model.s3(x)
        x = self.model.s4(x)
        x = self.model.final_conv(x)
        x = self.model.head.global_pool(x)
        x = self.lin(x)
        return x
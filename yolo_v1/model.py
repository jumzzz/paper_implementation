import torch
import torch.nn.functional as F

import torch.nn as nn
import torchvision.models as models


class YOLO_V1(nn.Module):
    
    def __init__(self,
                      *args, **kwargs):
        super(YOLO_V1, self).__init__(*args, **kwargs)
        
        self.dout00 = nn.Dropout(0.1)
        self.backbone = models.mobilenet_v2(pretrained=True).features
        self.dout01 = nn.Dropout(0.1)

        self.bneck_bn01 = nn.BatchNorm2d(1280)

        for p in self.backbone.parameters():
            p.requires_grad = False
        
        self.bneck_conv_01 = nn.Conv2d(1280, 
                                     1024, 
                                     kernel_size=1, 
                                     stride=2)
        
        self.dout02 = nn.Dropout(0.1)        
        self.bneck_bn02 = nn.BatchNorm2d(1024)
        self.bneck_flatten = nn.Flatten()
  
        self.dout03 = nn.Dropout(0.1)
        self.bneck_output = nn.Linear(7*7*1024, 1470)
        
        
    def forward(self, x):
        
        if self.training:
            x = self.dout00(x)
        x = self.backbone(x)
        
        if self.training:
            x = self.dout01(x)
        
        x = F.leaky_relu(self.bneck_bn01(x), negative_slope=0.1)
        
        x = self.bneck_conv_01(x)
        
        if self.training:
            x = self.dout02(x)
        
        x = F.leaky_relu(self.bneck_bn02(x), negative_slope=0.1)
        x = F.leaky_relu(self.bneck_flatten(x), negative_slope=0.1)
        
        if self.training:
            x = self.dout03(x)
        
        x = torch.sigmoid(self.bneck_output(x))
        
        return x.view(-1,30,7,7)

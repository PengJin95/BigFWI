#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from .block import *

# FlatFault/CurveFault
# 1000, 70 -> 70, 70
class FCN4_Deep_2(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(FCN4_Deep_2, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)

        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70) 
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35) 
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18) 
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9) 7
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock8(x) # (None, 512, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10) 
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20) 32, 28
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40) 64, 56
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80) 128, 112
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x



# FlatFault/CurveFault
# 1000, 70 -> 70, 70
class FCN4_Deep_2L(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(FCN4_Deep_2L, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock5_3 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock6_3 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock7_3 = ConvBlock(dim4, dim4)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)

        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv1_3 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv2_3 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv3_3 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv4_3 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv5_3 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70) 
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35) 
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock5_3(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18) 
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock6_3(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9) 7
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock7_3(x) # (None, 256, 8, 9)
        x = self.convblock8(x) # (None, 512, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv1_3(x) # (None, 512, 5, 5)

        x = self.deconv2_1(x) # (None, 256, 10, 10) 
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv2_3(x) # (None, 256, 10, 10)

        x = self.deconv3_1(x) # (None, 128, 20, 20) 32, 28
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv3_3(x) # (None, 128, 20, 20)

        x = self.deconv4_1(x) # (None, 64, 40, 40) 64, 56
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv4_3(x) # (None, 64, 40, 40)

        x = self.deconv5_1(x) # (None, 32, 80, 80) 128, 112
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = self.deconv5_3(x) # (None, 32, 80, 80)

        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x

# FlatFault/CurveFault
# 1000, 70 -> 70, 70
class FCN4_Deep_2XL(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, dim6=1024, sample_spatial=1.0, **kwargs):
        super(FCN4_Deep_2XL, self).__init__()
        self.convblock1 = ConvBlock(5, dim2, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim3, dim4, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim4, dim4, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim4, dim5, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim5, dim5, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim5, dim5, stride=2)
        self.convblock5_2 = ConvBlock(dim5, dim5)
        self.convblock6_1 = ConvBlock(dim5, dim5, stride=2)
        self.convblock6_2 = ConvBlock(dim5, dim5)
        self.convblock7_1 = ConvBlock(dim5, dim5, stride=2)
        self.convblock7_2 = ConvBlock(dim5, dim5)
        self.convblock8_1 = ConvBlock(dim5, dim6, stride=2)
        self.convblock8_2 = ConvBlock(dim6, dim6)
        self.convblock9 = ConvBlock(dim6, dim6, kernel_size=(4, 5), padding=0)

        self.deconv1_1 = DeconvBlock(dim6, dim6, kernel_size=2)
        self.deconv1_2 = ConvBlock(dim6, dim6)
        self.deconv2_1 = DeconvBlock(dim6, dim6, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim6, dim6)
        self.deconv3_1 = DeconvBlock(dim6, dim5, kernel_size=5, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim5, dim5)
        self.deconv4_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim4, dim4)
        self.deconv5_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim3, dim3)
        self.deconv6_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv6_2 = ConvBlock(dim2, dim2)
        self.deconv7 = ConvBlock_Tanh(dim2, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 64, 500, 70)
        x = self.convblock2_1(x) # (None, 128, 250, 70)
        x = self.convblock2_2(x) # (None, 128, 250, 70)
        x = self.convblock3_1(x) # (None, 256, 125, 70)
        x = self.convblock3_2(x) # (None, 256, 125, 70)
        x = self.convblock4_1(x) # (None, 512, 63, 70) 
        x = self.convblock4_2(x) # (None, 512, 63, 70)
        x = self.convblock5_1(x) # (None, 512, 32, 35) 
        x = self.convblock5_2(x) # (None, 512, 32, 35)
        x = self.convblock6_1(x) # (None, 512, 16, 18) 
        x = self.convblock6_2(x) # (None, 512, 16, 18)
        x = self.convblock7_1(x) # (None, 512, 8, 9)
        x = self.convblock7_2(x) # (None, 512, 8, 9)
        x = self.convblock8_1(x) # (None, 1024, 4, 5)
        x = self.convblock8_2(x) # (None, 1024, 4, 5)
        x = self.convblock9(x) # (None, 1024, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 1024, 2, 2)
        x = self.deconv1_2(x) # (None, 1024, 2, 2)
        x = self.deconv2_1(x) # (None, 1024, 4, 4) 
        x = self.deconv2_2(x) # (None, 1024, 4, 4)
        x = self.deconv3_1(x) # (None, 512, 9, 9) 
        x = self.deconv3_2(x) # (None, 512, 9, 9)
        x = self.deconv4_1(x) # (None, 256, 18, 18) 
        x = self.deconv4_2(x) # (None, 256, 18, 18)
        x = self.deconv5_1(x) # (None, 128, 36, 36)
        x = self.deconv5_2(x) # (None, 128, 36, 36)
        x = self.deconv6_1(x) # (None, 64, 72, 72)
        x = self.deconv6_2(x) # (None, 64, 72, 72)
        x = F.pad(x, [-1, -1, -1, -1], mode="constant", value=0) # (None, 64, 70, 70)
        x = self.deconv7(x) # (None, 1, 70, 70)
        return x


# FlatFault/CurveFault
# 1000, 7 -> 70, 70
class FCN4_Deep_2XL_2(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, dim6=1024, sample_spatial=1.0, **kwargs):
        super(FCN4_Deep_2XL_2, self).__init__()
        self.convblock1 = ConvBlock(5, dim2, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim3, dim4, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim4, dim4, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim4, dim5, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim5, dim5, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim5, dim5, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock5_2 = ConvBlock(dim5, dim5, kernel_size=(3, 1), padding=(1, 0))
        self.convblock6_1 = ConvBlock(dim5, dim5, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock6_2 = ConvBlock(dim5, dim5, kernel_size=(3, 1), padding=(1, 0))
        self.convblock7_1 = ConvBlock(dim5, dim5, stride=2)
        self.convblock7_2 = ConvBlock(dim5, dim5)
        self.convblock8_1 = ConvBlock(dim5, dim6, stride=2)
        self.convblock8_2 = ConvBlock(dim6, dim6)
        self.convblock9 = ConvBlock(dim6, dim6, kernel_size=(4, 2), padding=0)

        self.deconv1_1 = DeconvBlock(dim6, dim6, kernel_size=2)
        self.deconv1_2 = ConvBlock(dim6, dim6)
        self.deconv2_1 = DeconvBlock(dim6, dim6, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim6, dim6)
        self.deconv3_1 = DeconvBlock(dim6, dim5, kernel_size=5, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim5, dim5)
        self.deconv4_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim4, dim4)
        self.deconv5_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim3, dim3)
        self.deconv6_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv6_2 = ConvBlock(dim2, dim2)
        self.deconv7 = ConvBlock_Tanh(dim2, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 64, 500, 7)
        x = self.convblock2_1(x) # (None, 128, 250, 7)
        x = self.convblock2_2(x) # (None, 128, 250, 7)
        x = self.convblock3_1(x) # (None, 256, 125, 7)
        x = self.convblock3_2(x) # (None, 256, 125, 7)
        x = self.convblock4_1(x) # (None, 512, 63, 7) 
        x = self.convblock4_2(x) # (None, 512, 63, 7)
        x = self.convblock5_1(x) # (None, 512, 32, 7) 
        x = self.convblock5_2(x) # (None, 512, 32, 7)
        x = self.convblock6_1(x) # (None, 512, 16, 7) 
        x = self.convblock6_2(x) # (None, 512, 16, 7)
        x = self.convblock7_1(x) # (None, 512, 8, 4)
        x = self.convblock7_2(x) # (None, 512, 8, 4)
        x = self.convblock8_1(x) # (None, 1024, 4, 2)
        x = self.convblock8_2(x) # (None, 1024, 4, 2)
        x = self.convblock9(x) # (None, 1024, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 1024, 2, 2)
        x = self.deconv1_2(x) # (None, 1024, 2, 2)
        x = self.deconv2_1(x) # (None, 1024, 4, 4) 
        x = self.deconv2_2(x) # (None, 1024, 4, 4)
        x = self.deconv3_1(x) # (None, 512, 9, 9) 
        x = self.deconv3_2(x) # (None, 512, 9, 9)
        x = self.deconv4_1(x) # (None, 256, 18, 18) 
        x = self.deconv4_2(x) # (None, 256, 18, 18)
        x = self.deconv5_1(x) # (None, 128, 36, 36)
        x = self.deconv5_2(x) # (None, 128, 36, 36)
        x = self.deconv6_1(x) # (None, 64, 72, 72)
        x = self.deconv6_2(x) # (None, 64, 72, 72)
        x = F.pad(x, [-1, -1, -1, -1], mode="constant", value=0) # (None, 64, 70, 70)
        x = self.deconv7(x) # (None, 1, 70, 70)
        return x



if __name__ == '__main__':
    device = torch.device('cpu')
    model = FCN4_Deep_2() # 24M: 12+12
    # model = FCN4_Deep_2L() # 29M
    # model = FCN4_Deep_2XL() # 102M=51+58
    # model = FCN4_Deep_2XL_2() # 87M
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters: %d' % total_params)
    x = torch.rand((3, 5, 1000, 7))
    y = model(x)
    print(y.shape)

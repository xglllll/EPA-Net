from email.mime import base
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res):
        super(EBlock, self).__init__()

        layers = [ConvFFN(out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res):
        super(DBlock, self).__init__()

        layers = [ConvFFN(channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class EBlock_v1(nn.Module):
    def __init__(self, channel, reduction, dmff_reduction, num_head, num_res):
        super(EBlock_v1, self).__init__()

        self.head = LPAB_head(channel=channel, reduction=reduction, dmff_reduction=dmff_reduction, num_head=num_head)
        self.layer = [LPAB_mid(channel=channel, reduction=reduction, dmff_reduction=dmff_reduction, num_head=num_head) for _ in range(num_res-2)]

        self.layers = nn.Sequential(*self.layer)
        self.tail = LPAB_tail(channel=channel, reduction=reduction, dmff_reduction=dmff_reduction, num_head=num_head)

    def forward(self, x):
        x = self.tail(self.layers(self.head(x)))
        return x

    def flops(self, H, W):

        flop = 0

        flop += self.head.flops(H, W)

        for layer in self.layer:
            flop += layer.flops(H, W)

        flop += self.tail.flops(H, W)

        return flop


class DBlock_v1(nn.Module):
    def __init__(self, channel, reduction, dmff_reduction, num_head, num_res):
        super(DBlock_v1, self).__init__()

        self.head = LPAB_head(channel=channel, reduction=reduction, dmff_reduction=dmff_reduction, num_head=num_head)
        self.layer = [LPAB_mid(channel=channel, reduction=reduction, dmff_reduction=dmff_reduction, num_head=num_head) for _ in range(num_res-2)]

        self.layers = nn.Sequential(*self.layer)
        self.tail = LPAB_tail(channel=channel, reduction=reduction, dmff_reduction=dmff_reduction, num_head=num_head)

    def forward(self, x):
        x = self.tail(self.layers(self.head(x)))
        return x

    def flops(self, H, W):

        flop = 0

        flop += self.head.flops(H, W)

        for layer in self.layer:
            flop += layer.flops(H, W)

        flop += self.tail.flops(H, W)

        return flop

class LPANet(nn.Module):
    def __init__(self, channel=48, reduction=4, dmff_reduction=16, num_heads=[1,2,4,8], num_res=[2,4,12,3]):
        super(LPANet, self).__init__()

        self.channel = channel

        self.Encoder = nn.ModuleList([
            EBlock_v1(channel=channel,   reduction=reduction, dmff_reduction=dmff_reduction, num_head=num_heads[0], num_res=num_res[0]),
            EBlock_v1(channel=channel*2, reduction=reduction, dmff_reduction=dmff_reduction, num_head=num_heads[1], num_res=num_res[1]),
            EBlock_v1(channel=channel*4, reduction=reduction, dmff_reduction=dmff_reduction, num_head=num_heads[2], num_res=num_res[2]),
            EBlock_v1(channel=channel*8, reduction=reduction, dmff_reduction=dmff_reduction, num_head=num_heads[3], num_res=num_res[3]),
        ])

        self.downsample = nn.ModuleList([
            BasicConv(channel,   channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(channel*2, channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(channel*4, channel*8, kernel_size=3, relu=True, stride=2)

        ])

        self.Decoder = nn.ModuleList([
            DBlock_v1(channel=channel*8, reduction=reduction, dmff_reduction=dmff_reduction, num_head=num_heads[3], num_res=num_res[3]),
            DBlock_v1(channel=channel*4, reduction=reduction, dmff_reduction=dmff_reduction, num_head=num_heads[2], num_res=num_res[2]),
            DBlock_v1(channel=channel*2, reduction=reduction, dmff_reduction=dmff_reduction, num_head=num_heads[1], num_res=num_res[1]),
            DBlock_v1(channel=channel,   reduction=reduction, dmff_reduction=dmff_reduction, num_head=num_heads[0], num_res=num_res[0]),
        ])
        
        self.upsample = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel*8, channel*16, 1, bias=False),
                nn.PixelShuffle(2),
            ),

            nn.Sequential(
                nn.Conv2d(channel*4, channel*8, 1, bias=False),
                nn.PixelShuffle(2),
            ),

            nn.Sequential(
                nn.Conv2d(channel*2, channel*4, 1, bias=False),
                nn.PixelShuffle(2),
            )      
        ])
        

        self.Convs = nn.ModuleList([
            BasicConv(3, channel, kernel_size=3, relu=True,  stride=1),
            BasicConv(channel, 3, kernel_size=3, relu=False, stride=1)
        ])


    def forward(self, x):

        z = self.Convs[0](x)#3->32
        res1 = self.Encoder[0](z)#32->32, 32,h

        z = self.downsample[0](res1)#32->64,h->h/2
        res2 = self.Encoder[1](z)#64->64,h/2

        z = self.downsample[1](res2)#64->128,h/2->h/4
        res3 = self.Encoder[2](z)#128->128,h/4

        z = self.downsample[2](res3)#128->256,h/4->h/8
        res4 = self.Encoder[3](z)#256->256,h/8

        out1 = self.Decoder[0](res4)#256->256
        z = self.upsample[0](out1)#256->128,h/8->h/4

        z = z + res3#h/4,128
        out2 = self.Decoder[1](z)#h/4,128->128

        z = self.upsample[1](out2)#h/2,128->64
        
        z = z + res2#h/2,64
        out3 = self.Decoder[2](z)#h/2,64->64

        z= self.upsample[2](out3)#h,64->32

        z = z + res1#h,32
        z = self.Decoder[3](z)#32->32

        z = self.Convs[1](z)#32->3

        return z+x
    
    def flops(self, x):

        _, _, H, W = x.shape

        flop = 0
        
        flop += self.Encoder[0].flops(H, W)

        flop += self.Encoder[1].flops(H / 2, W / 2)
        
        flop += self.Encoder[2].flops(H / 4, W / 4)
        
        flop += self.Encoder[3].flops(H / 8, W / 8)

        flop += self.Decoder[0].flops(H / 8, W / 8)

        flop += self.Decoder[1].flops(H / 4, W / 4)

        flop += self.Decoder[2].flops(H / 2, W / 2)

        flop += self.Decoder[3].flops(H, W)


        return flop  

import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.LeakyReLU(0.2,inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class Attention(nn.Module):
    def __init__(self, channel, num_head):
        super(Attention, self).__init__()
        self.channel = channel
        self.num_head = num_head
        self.head_dim = channel // num_head
        self.scale = self.head_dim ** -0.5

        # self.ln = LayerNorm2d(channel)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_kv = nn.Conv2d(channel, channel*2, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, C, H, W = x.shape

        shortcut = x

        x_q = self.avg_pool(x) # B, C, 1, 1

        k_v = self.conv_kv(x_q) # B, C//2, 1, 1

        k, v = torch.chunk(k_v, chunks=2, dim=1) # B, C//4, 1, 1

        # print(x_q.shape, k.shape, v.shape)
        q = rearrange(x_q, 'b (n c) h w -> b n c (h w)', n=self.num_head) # b, n, c, 1
        k = rearrange(k,   'b (n c) h w -> b n c (h w)', n=self.num_head) # b, n, c//4, 1
        v = rearrange(v,   'b (n c) h w -> b n c (h w)', n=self.num_head) # b, n, c//4, 1

        attn = (q @ k.transpose(-2, -1)) * self.scale # b, n, c, c//4
        attn = attn.softmax(dim=-1)
        out = attn @ v # b, n, c, (1 1)
        out = rearrange(out, 'b n c (h w)-> b (n c) h w', h=1) # (win_num*b), c, win_h, win_w
        out = self.sigmoid(out)
        out = out * shortcut + shortcut

        return out

    def flops(self, H, W):
        # b, c, h, w = x.shape
        b = 4

        flop = 0

        # q @ k^T
        flop += b * (self.channel // self.num_head) * self.channel * 1 * self.channel // 4
        # atten @ v
        flop += b * (self.channel // self.num_head) * self.channel * self.channel // 4 * 1

        return flop 


# class DMFF(nn.Module):
#     def __init__(self, channel, reduction = 16):
#         super(DMFF,self).__init__()
#         self.channel = channel
#         self.reduction = reduction
#         self.conv_down = nn.Conv2d(channel, channel // reduction, kernel_size=1, bias = False)
#         self.conv_up1  = nn.Conv2d(channel // reduction, channel, kernel_size=1, bias = False)
#         self.conv_up2  = nn.Conv2d(channel // reduction, channel, kernel_size=1, bias = False)
#         self.relu = nn.ReLU(True)
#         self.sigmoid = nn.Sigmoid()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)

#     def forward(self, x):
#         x1 = x[0]
#         x2 = x[1]
#         x3 = x1 + x2
#         x4 = self.avg_pool(x3)
#         x5 = self.conv_down(x4)
#         x6 = self.relu(x5)
#         x6_1 = self.conv_up1(x6)
#         x6_sig1 = self.sigmoid(x6_1)
#         x6_out1 = torch.mul(x1,x6_sig1)

#         x6_2 = self.conv_up2(x6)
#         x6_sig2 = self.sigmoid(x6_2)
#         x6_out2 = torch.mul(x2,x6_sig2)
#         x7 = x6_out1 + x6_out2
#         x8 = x3 + x7
#         return x8

class DMFF(nn.Module):
    def __init__(self, channel=48, dmff_reduction=16, num_head=1):
        super(DMFF,self).__init__()

        self.channel = channel
        self.num_head = num_head
        self.scale = channel ** -0.5
        self.conv_k  = nn.Conv2d(channel, channel // dmff_reduction, 1, bias = False)
        self.conv_v1 = nn.Conv2d(channel, channel // dmff_reduction, 1, bias = False)
        self.conv_v2 = nn.Conv2d(channel, channel // dmff_reduction, 1, bias = False)

        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        x_12 = x1 + x2

        x_12_pool = self.avg_pool(x_12)
        k = self.conv_k(x_12_pool) # b, c//16, 1, 1
        k = rearrange(k, 'b (n c) h w -> b n c (h w)', n=self.num_head) # b, n, c // 16, 1

        q1 = self.avg_pool(x1) # b, c, 1, 1
        v1 = self.conv_v1(q1) # b, c//16, 1, 1

        q1 = rearrange(q1, 'b (n c) h w -> b n c (h w)', n=self.num_head) # b, n, c, 1
        v1 = rearrange(v1, 'b (n c) h w -> b n c (h w)', n=self.num_head) # b, n, c // 16, 1

        attn1 = (q1 @ k.transpose(-2, -1)) * self.scale # b, n, c, c//16
        attn1 = attn1.softmax(dim=-1)
        out1 = attn1 @ v1 # b, n, c, (1 1)
        out1 = rearrange(out1, 'b n c (h w)-> b (n c) h w', h=1) # (win_num*b), c, win_h, win_w

        out1 = self.sigmoid(out1)
        out1 = torch.mul(x1, out1)

        q2 = self.avg_pool(x2) # b, c, 1, 1
        v2 = self.conv_v2(q2) # b, c//16, 1, 1

        q2 = rearrange(q2, 'b (n c) h w -> b n c (h w)', n=self.num_head)
        v2 = rearrange(v2, 'b (n c) h w -> b n c (h w)', n=self.num_head)

        attn2 = (q2 @ k.transpose(-2, -1)) * self.scale # b, n, c, c//16
        attn2 = attn2.softmax(dim=-1)
        out2 = attn2 @ v2 # b, n, c, (1 1)
        out2 = rearrange(out2, 'b n c (h w)-> b (n c) h w', h=1) # (win_num*b), c, win_h, win_w

        out2 = self.sigmoid(out2)
        out2 = torch.mul(x2, out2)
        out = out1 + out2

        out = x_12 + out

        return out

    def flops(self, H, W):
        # b, c, h, w = x.shape
        b = 4

        flop = 0

        # q1 @ k^T
        flop += b * (self.channel // self.num_head) * self.channel * 1 * self.channel // 16
        # atten1 @ v1
        flop += b * (self.channel // self.num_head) * self.channel * self.channel // 16 * 1

        # q2 @ k^T
        flop += b * (self.channel // self.num_head) * self.channel * 1 * self.channel // 16
        # atten2 @ v2
        flop += b * (self.channel // self.num_head) * self.channel * self.channel // 16 * 1

        return flop 


class LPAM_head(nn.Module):
    def __init__(self, channel=48, reduction=4, dmff_reduction=16, num_head=1):
        super(LPAM_head,self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.dmff_reduction = dmff_reduction

        # self.lcfe = Attention(channel, num_head)

        self.lcfe = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, kernel_size=1, bias=False),
            nn.ReLU(True),
            Attention(channel//reduction, num_head),
            nn.Conv2d(channel//reduction, channel, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

        self.dmff = nn.ModuleList([
            DMFF(channel=channel, dmff_reduction=dmff_reduction, num_head=num_head), 
            DMFF(channel=channel, dmff_reduction=dmff_reduction, num_head=num_head), 
            DMFF(channel=channel, dmff_reduction=dmff_reduction, num_head=num_head), 
        ])
        
        self.depth = nn.ModuleList([
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
        ])
    
    def forward(self,x):
        x1 = self.lcfe(x)# 1
        x1_1 = F.interpolate(x1, scale_factor=0.5,   recompute_scale_factor=True, mode='area')# 1/2
        x1_2 = F.interpolate(x1, scale_factor=0.25,  recompute_scale_factor=True, mode='area')# 1/4
        x1_3 = F.interpolate(x1, scale_factor=0.125, recompute_scale_factor=True, mode='area')# 1/8

        x1_2_depth = self.depth[0](x1_2) # 1/4
        x1_3_depth = self.depth[1](x1_3) # upsample after depth
        x1_3_up = F.interpolate(x1_3_depth, scale_factor=2, recompute_scale_factor=True) # 1/4
        x1_2_fuse = self.dmff[0]([x1_2_depth, x1_3_up])

        x1_1_depth = self.depth[2](x1_1) # 1/2
        x1_2_fuse_depth = self.depth[3](x1_2_fuse)
        x1_2_fuse_up = F.interpolate(x1_2_fuse_depth, scale_factor=2, recompute_scale_factor=True) # 1/2
        x1_1_fuse = self.dmff[1]([x1_1_depth, x1_2_fuse_up])

        x1_depth = self.depth[4](x1)# 1
        x1_1_fuse_depth = self.depth[5](x1_1_fuse)
        x1_1_fuse_up = F.interpolate(x1_1_fuse_depth, scale_factor=2, recompute_scale_factor=True) # 1
        x1_fuse = self.dmff[2]([x1_depth, x1_1_fuse_up])

        attn = self.sigmoid(x1_fuse)
        out = torch.mul(x, attn)
        return out, x1_fuse

    def flops(self, H, W):

        flop = 0

        # flop += self.lcfe.flops(H, W)

        # flop += self.dmff[0].flops(H, W)

        # flop += self.dmff[1].flops(H, W)
        
        # flop += self.dmff[2].flops(H, W)

        return flop

class LPAM_mid(nn.Module):
    def __init__(self, channel=48, reduction=4, dmff_reduction=16, num_head=1):
        super(LPAM_mid, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.dmff_reduction = dmff_reduction

        # self.lcfe = Attention(channel, num_head)
        self.lcfe = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, kernel_size=1, bias=False),
            nn.ReLU(True),
            Attention(channel//reduction, num_head),
            nn.Conv2d(channel//reduction, channel, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        self.dmff = nn.ModuleList([
            DMFF(channel=channel, dmff_reduction=dmff_reduction, num_head=num_head), 
            DMFF(channel=channel, dmff_reduction=dmff_reduction, num_head=num_head), 
            DMFF(channel=channel, dmff_reduction=dmff_reduction, num_head=num_head), 
        ])

        self.depth = nn.ModuleList([
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
        ])
    
    def forward(self,x):
        x0 = x[0]
        feature_last = x[1]
        x1 = self.lcfe(x0)# 1
        x1_1 = F.interpolate(x1, scale_factor=0.5,   recompute_scale_factor=True, mode='area')# 1/2
        x1_2 = F.interpolate(x1, scale_factor=0.25,  recompute_scale_factor=True, mode='area')# 1/4
        x1_3 = F.interpolate(x1, scale_factor=0.125, recompute_scale_factor=True, mode='area')# 1/8

        x1_2_depth = self.depth[0](x1_2) # 1/4
        x1_3_depth = self.depth[1](x1_3) # upsample after depth
        x1_3_up = F.interpolate(x1_3_depth, scale_factor=2, recompute_scale_factor=True) # 1/4
        x1_2_fuse = self.dmff[0]([x1_2_depth, x1_3_up])

        x1_1_depth = self.depth[2](x1_1) # 1/2
        x1_2_fuse_depth = self.depth[3](x1_2_fuse)
        x1_2_fuse_up = F.interpolate(x1_2_fuse_depth, scale_factor=2, recompute_scale_factor=True) # 1/2
        x1_1_fuse = self.dmff[1]([x1_1_depth,x1_2_fuse_up])

        x1_depth = self.depth[4](x1)# 1
        x1_1_fuse_depth = self.depth[5](x1_1_fuse)
        x1_1_fuse_up = F.interpolate(x1_1_fuse_depth, scale_factor=2, recompute_scale_factor=True) # 1
        x1_fuse = self.dmff[2]([x1_depth, x1_1_fuse_up])

        x_cross = x1_fuse + feature_last
        attn = self.sigmoid(x_cross)
        out = torch.mul(x0, attn)
        return out, x_cross

    def flops(self, H, W):

        flop = 0

        # flop += self.lcfe.flops(H, W)

        # flop += self.dmff[0].flops(H, W)

        # flop += self.dmff[1].flops(H, W)
        
        # flop += self.dmff[2].flops(H, W)

        return flop

class LPAM_tail(nn.Module):
    def __init__(self, channel=48, reduction=4, dmff_reduction=16, num_head=1):
        super(LPAM_tail,self).__init__()
        self.channel = channel
        self.reduction = reduction
        # self.lcfe = nn.Sequential(
        #     nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
        #     nn.ReLU(True),
        #     nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False)
        # )

        # self.lcfe = Attention(channel, num_head)

        self.lcfe = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, kernel_size=1, bias=False),
            nn.ReLU(True),
            Attention(channel//reduction, num_head),
            nn.Conv2d(channel//reduction, channel, kernel_size=1, bias=False)
        )

        # self.dmff1 = DMFF(dim)
        # self.dmff2 = DMFF(dim)
        # self.dmff3 = DMFF(dim)
        self.sigmoid = nn.Sigmoid()

        self.dmff = nn.ModuleList([
            DMFF(channel=channel, dmff_reduction=dmff_reduction, num_head=num_head), 
            DMFF(channel=channel, dmff_reduction=dmff_reduction, num_head=num_head), 
            DMFF(channel=channel, dmff_reduction=dmff_reduction, num_head=num_head), 
        ])
        

        self.depth = nn.ModuleList([
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
        ])

        # self.depth1 = nn.Conv2d(dim,dim,kernel_size=3, padding=1,groups=dim,bias=False)
        # self.depth2 = nn.Conv2d(dim,dim,kernel_size=3, padding=1,groups=dim,bias=False)
        # self.depth3 = nn.Conv2d(dim,dim,kernel_size=3, padding=1,groups=dim,bias=False)
        # self.depth4 = nn.Conv2d(dim,dim,kernel_size=3, padding=1,groups=dim,bias=False)
        # self.depth5 = nn.Conv2d(dim,dim,kernel_size=3, padding=1,groups=dim,bias=False)
        # self.depth6 = nn.Conv2d(dim,dim,kernel_size=3, padding=1,groups=dim,bias=False)

    
    def forward(self,x):
        x0 = x[0]
        feature_last = x[1]
        x1 = self.lcfe(x0)# 1
        x1_1 = F.interpolate(x1, scale_factor=0.5,   recompute_scale_factor=True, mode='area')# 1/2
        x1_2 = F.interpolate(x1, scale_factor=0.25,  recompute_scale_factor=True, mode='area')# 1/4
        x1_3 = F.interpolate(x1, scale_factor=0.125, recompute_scale_factor=True, mode='area')# 1/8

        x1_2_depth = self.depth[0](x1_2) # 1/4
        x1_3_depth = self.depth[1](x1_3) # upsample after depth
        x1_3_up = F.interpolate(x1_3_depth, scale_factor=2, recompute_scale_factor=True) # 1/4
        x1_2_fuse = self.dmff[0]([x1_2_depth, x1_3_up])

        x1_1_depth = self.depth[2](x1_1) # 1/2
        x1_2_fuse_depth = self.depth[3](x1_2_fuse)
        x1_2_fuse_up = F.interpolate(x1_2_fuse_depth, scale_factor=2, recompute_scale_factor=True) # 1/2
        x1_1_fuse = self.dmff[1]([x1_1_depth, x1_2_fuse_up])

        x1_depth = self.depth[4](x1)# 1
        x1_1_fuse_depth = self.depth[5](x1_1_fuse)
        x1_1_fuse_up = F.interpolate(x1_1_fuse_depth, scale_factor=2, recompute_scale_factor=True) # 1
        x1_fuse = self.dmff[2]([x1_depth, x1_1_fuse_up])

        x_cross = x1_fuse + feature_last
        attn = self.sigmoid(x_cross)
        out = torch.mul(x0, attn)
        return out

    def flops(self, H, W):

        flop = 0

        # flop += self.lcfe.flops(H, W)

        # flop += self.dmff[0].flops(H, W)

        # flop += self.dmff[1].flops(H, W)
        
        # flop += self.dmff[2].flops(H, W)

        return flop

class CA_Layer(nn.Module):
    def __init__(self,dim):
        super(CA_Layer,self).__init__()
        self.dim = dim
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim,dim//2,1,bias=False),
            nn.PReLU(),
            nn.Conv2d(dim//2,dim,1,bias=False),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        y = self.ca(x)
        return y * x

class LPAB_head(nn.Module):
    def __init__(self, channel, reduction, dmff_reduction, num_head):
        super(LPAB_head,self).__init__()
        self.channel = channel
        self.norm1 = LayerNorm2d(channel)
        self.norm2 = LayerNorm2d(channel)
        self.lpam = LPAM_head(channel=channel, reduction=reduction, dmff_reduction=dmff_reduction, num_head=num_head)
        self.convffn = nn.Sequential(
            nn.Conv2d(channel, channel*4, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channel*4, channel*4, kernel_size=3, padding=1, groups=channel*4, bias=False),
            nn.GELU(),
            nn.Conv2d(channel*4, channel, kernel_size=1, bias=False)
        )
        # self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        inp = x
        x, feature = self.lpam(x)
        x = x + inp
        y = self.norm1(x)
        x = self.convffn(y) + y
        y = self.norm2(x)
        return y, feature

    def flops(self, H, W):

        flops = 0

        flops += self.lpam.flops(H, W)

        return flops

class LPAB_mid(nn.Module):
    def __init__(self, channel, reduction, dmff_reduction, num_head):
        super(LPAB_mid, self).__init__()
        self.channel = channel
        self.norm1 = LayerNorm2d(channel)
        self.norm2 = LayerNorm2d(channel)
        self.lpam = LPAM_mid(channel=channel, reduction=reduction, dmff_reduction=dmff_reduction, num_head=num_head)
        self.convffn = nn.Sequential(
            nn.Conv2d(channel, channel*4, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channel*4, channel*4, kernel_size=3, padding=1, groups=channel*4, bias=False),
            nn.GELU(),
            nn.Conv2d(channel*4, channel,kernel_size=1,bias=False)
        )
        # self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        inp = x[0]
        feature = x[1]
        x1, feature1 = self.lpam([inp, feature])
        x2 = x1 + inp
        y = self.norm1(x2)
        x2 = self.convffn(y) + y
        y = self.norm2(x2)
        return y, feature1

    
    def flops(self, H, W):

        flops = 0

        flops += self.lpam.flops(H, W)

        return flops

class LPAB_tail(nn.Module):
    def __init__(self, channel, reduction, dmff_reduction, num_head):
        super(LPAB_tail,self).__init__()
        self.channel = channel
        self.norm1 = LayerNorm2d(channel)
        self.norm2 = LayerNorm2d(channel)
        self.lpam = LPAM_tail(channel=channel, reduction=reduction, dmff_reduction=dmff_reduction, num_head=num_head)
        self.convffn = nn.Sequential(
            nn.Conv2d(channel, channel*4, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channel*4, channel*4, kernel_size=3, padding=1, groups=channel*4, bias=False),
            nn.GELU(),
            nn.Conv2d(channel*4, channel, kernel_size=1, bias=False)
        )
        # self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self,x):
        inp = x[0]
        feature = x[1]
        x1 = self.lpam([inp, feature])
        x2 = x1 + inp
        y = self.norm1(x2)
        x2 = self.convffn(y) + y
        y = self.norm2(x2)
        return y

    
    def flops(self, H, W):

        flops = 0

        flops += self.lpam.flops(H, W)

        return flops

class ConvFFN(nn.Module):
    def __init__(self, channel):
        super(ConvFFN,self).__init__()
        self.channel = channel
        self.norm = LayerNorm2d(channel)
        self.convffn = nn.Sequential(
            nn.Conv2d(channel, channel*4, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channel*4, channel*4, kernel_size=3, padding=1, groups=channel*4, bias=False),
            nn.GELU(),
            nn.Conv2d(channel*4, channel, kernel_size=1, bias=False)
        )

    def forward(self,x):
        x = x + self.convffn(self.norm(x))
        return x

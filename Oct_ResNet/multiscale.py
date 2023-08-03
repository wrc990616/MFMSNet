
import torch

import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel,kernel_size, stride=1,padding=0, dilation=1,groups=1,act='relu'):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
        )
        if act is None:
            del self.conv[-1]
    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class TransposeX2(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        layers = [
            #nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),

            nn.ConvTranspose2d(in_channels, out_channels,kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = False),
            nn.ReLU(inplace=True),
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels))

        super().__init__(*layers)

import math

class EfficientChannelAttention(nn.Module):              # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x
        x = self.avg_pool(x)
        x = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(x)
        return out*y

class CBAM(nn.Module):
    def __init__(self,in_channel,reduction=16,kernel_size=7):
        super(CBAM, self).__init__()
        #通道注意力机制
        self.max_pool=nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp=nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction,out_features=in_channel,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
        #空间注意力机制
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=kernel_size ,stride=1,padding=kernel_size//2,bias=False)

    def forward(self,x):
        #通道注意力机制
        maxout=self.max_pool(x)
        maxout=self.mlp(maxout.view(maxout.size(0),-1))
        avgout=self.avg_pool(x)
        avgout=self.mlp(avgout.view(avgout.size(0),-1))
        channel_out=self.sigmoid(maxout+avgout)
        channel_out=channel_out.view(x.size(0),x.size(1),1,1)
        channel_out=channel_out*x
        #空间注意力机制
        max_out,_=torch.max(channel_out,dim=1,keepdim=True)
        mean_out=torch.mean(channel_out,dim=1,keepdim=True)
        out=torch.cat((max_out,mean_out),dim=1)
        out=self.sigmoid(self.conv(out))
        out=out*channel_out
        return out

class multiscale(nn.Module):
    def __init__(self,in_channels,out_channels,fin_channels):
        super(multiscale, self).__init__()
        self.conv1 = ConvBNReLU(in_channel=in_channels // 2,out_channel=out_channels,kernel_size=3,stride=2,padding=1)
        self.CA = ChannelAttention(in_planes=out_channels)
        #self.conv13 = ConvBNReLU(in_channel=in_channels,out_channel=out_channels,kernel_size=(1,3),padding=(0,1))
        #self.conv31 = ConvBNReLU(in_channel=out_channels, out_channel=out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.conv_1 = ConvBNReLU(in_channel=in_channels,out_channel=out_channels,kernel_size=1,padding=0)
        self.norm1 = LayerNorm(out_channels*3,eps=1e-6, data_format="channels_first")
        self.conv11 = nn.Conv2d(in_channels= out_channels*3,out_channels=out_channels,kernel_size=1)
        self.re = nn.ReLU(True)
        self.gelu = nn.GELU()

        self.conv_l_11 = ConvBNReLU(in_channel=in_channels*2,out_channel=out_channels,kernel_size=1)
        self.up = TransposeX2(in_channels=out_channels,out_channels=out_channels)
        self.SA = SpatialAttention(kernel_size=7)


        ###########
        self.conv11_2 = ConvBNReLU(in_channel=out_channels*3,out_channel=64,kernel_size=3,padding=1)
        #self.mlp = IRMLP(inp_dim=out_channels*3,out_dim=fin_channels)
        self.eca = EfficientChannelAttention(c=64)
        #self.eca = CBAM(in_channel=64)

        #self.up4 = nn.Sequential(TransposeX2(in_channels=out_channels,out_channels=out_channels),TransposeX2(in_channels=out_channels,out_channels=out_channels))
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(fin_channels, fin_channels, kernel_size=16, stride=8, padding=4),
            nn.BatchNorm2d(fin_channels),
            nn.ReLU()
        )


        self.up4 = nn.UpsamplingBilinear2d(scale_factor=8)
        #self.fin = nn.Conv2d(in_channels=out_channels,out_channels=fin_channels,kernel_size=1)



    def forward(self,h,m,l):
        h =self.conv1(h)
        h_att = self.SA(h)*h

        #m = self.conv31(self.conv13(m))
        m = self.conv_1(m)

        l = self.conv_l_11(l)
        l = self.up(l)
        l_att = l*self.CA(l)

        fusion = torch.cat((h,m,l),dim=1)
        fusion = self.norm1(fusion)
        fusion_1 = self.gelu(self.conv11(fusion))

        fusion_2 = torch.cat((h_att,fusion_1,l_att),dim=1)


        fusion_2 = self.conv11_2(fusion_2)
        #print(fusion_2.size())
        fusion_2 = self.eca(fusion_2)
        #fusion_2 = self.mlp(fusion_2)
        #fusion_2 = self.fin(fusion_2)
        out = self.up4(fusion_2)

        return out

#### Inverted Residual MLP
class IRMLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(IRMLP, self).__init__()
        self.conv1 = Conv(inp_dim, inp_dim, 1, relu=False, bias=False, group=inp_dim)
        self.conv2 = Conv(inp_dim, inp_dim //4, 1, relu=False, bias=False)
        self.conv3 = Conv(inp_dim //4, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(inp_dim)

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual

        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)

        return out

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x




if __name__=='__main__':
    x = torch.randn(1,3,256,256)
    model = ConvBNReLU(in_channel=3,out_channel=64,kernel_size=(1,3),padding=(0,1))
    y = model(x)
    print(y.size())
    inputs = torch.ones(1, 1, 64, 64)
    transposed_conv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=4, padding=0, output_padding=1,bias=False)
    outputs = transposed_conv(inputs)
    print(outputs.size())

    l = torch.randn(1, 512, 16, 16)
    m = torch.randn(1, 256, 32, 32)
    h = torch.randn(1, 128, 64, 64)

    model = multiscale(in_channels=256,out_channels=256,fin_channels=64)
    y = model(h,m,l)
    print('ok:', y.size())

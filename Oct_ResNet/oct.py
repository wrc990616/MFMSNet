import torch
import torch.nn as nn

from octconv import *
from muti_freq_transformer import  multi_Freq
from multiscale import multiscale
#from mymodel.Oct_ResNet.octconv import *




class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv_BN_ACT(inplanes, width, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out,
                                 norm_layer=norm_layer)
        self.conv2 = Conv_BN_ACT(width, width, kernel_size=3, stride=stride, padding=1, groups=groups,
                                 norm_layer=norm_layer,
                                 alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5)
        self.conv3 = Conv_BN(width, planes * self.expansion, kernel_size=1, norm_layer=norm_layer,
                             alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity_h = x[0] if type(x) is tuple else x
        identity_l = x[1] if type(x) is tuple else None

        x_h, x_l = self.conv1(x)
        x_h, x_l = self.conv2((x_h, x_l))
        x_h, x_l = self.conv3((x_h, x_l))

        if self.downsample is not None:
            identity_h, identity_l = self.downsample(x)

        x_h += identity_h
        x_l = x_l + identity_l if identity_l is not None else None

        x_h = self.relu(x_h)
        x_l = self.relu(x_l) if x_l is not None else None

        return x_h, x_l


class decode_4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decode_4, self).__init__()
        self.in_channels = in_channels
        self.conv = OCTConv_BN_ACT(in_channels=in_channels, out_channels=out_channels, kernel_size=3, alpha_in=0)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x = self.up(x)

        x_h, x_l = self.conv(x)

        return x_h, x_l


class decode(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decode, self).__init__()
        self.conv = OCTConv_BN_ACT(in_channels=in_channels, out_channels=out_channels, kernel_size=3)

    def forward(self, en_h, en_l, de_h, de_l):
        h = torch.cat((en_h, de_h), dim=1)
        l = torch.cat((en_l, de_l), dim=1)
        h = nn.UpsamplingBilinear2d(scale_factor=2)(h)
        l = nn.UpsamplingBilinear2d(scale_factor=2)(l)
        hh, ll = self.conv((h, l))
        return hh, ll


class decode_0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decode_0, self).__init__()
        self.conv = OCTConv_BN_ACT(in_channels=in_channels, out_channels=out_channels, kernel_size=3, alpha_out=0)

    def forward(self, en_h, en_l, de_h, de_l):
        h = torch.cat((en_h, de_h), dim=1)
        l = torch.cat((en_l, de_l), dim=1)
        h = nn.UpsamplingBilinear2d(scale_factor=2)(h)
        l = nn.UpsamplingBilinear2d(scale_factor=2)(l)
        hh, ll = self.conv((h, l))
        return hh, ll




class decode_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decode_1, self).__init__()
        self.conv = OCTConv_BN_ACT(in_channels=in_channels, out_channels=out_channels, kernel_size=3, alpha_out=0)

    def forward(self, en_h, en_l, de_h, de_l):
        h = torch.cat((en_h, de_h), dim=1)
        l = torch.cat((en_l, de_l), dim=1)
        h = nn.UpsamplingBilinear2d(scale_factor=2)(h)
        l = nn.UpsamplingBilinear2d(scale_factor=2)(l)
        hh, ll = self.conv((h, l))
        return hh, ll


class decode_00(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decode_00, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x, y):
        a = torch.cat((x, y), dim=1)
        a = self.conv(a)
        a = nn.UpsamplingBilinear2d(scale_factor=2)(a)
        return a

class double_oct(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,alpha_in=0.5,alpha_out=0.5):
        super(double_oct, self).__init__()
        self.alpha_in = alpha_in
        self.conv1 = OCTConv_BN_ACT(in_channels=in_channels,out_channels=in_channels//4,kernel_size=kernel_size,
                                    alpha_in=alpha_in,alpha_out=0.5)
        self.conv2 = OCTConv_BN_ACT(in_channels=in_channels//4, out_channels=out_channels, kernel_size=kernel_size,
                                    alpha_in=0.5, alpha_out=alpha_out)
    def forward(self,h,l):

        if self.alpha_in == 0:
            hh,ll = self.conv1(h)
            hh,ll = self.conv2((hh,ll))
        else:
            hh,ll = self.conv1((h,l))
            hh,ll =self.conv2((hh,ll))
        return hh,ll



class decode_all(nn.Module):
    def __init__(self, in_channels, out_channels,alpha_in=0.5,alpha_out=0.5):
        super(decode_all, self).__init__()
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        if alpha_out==0:
            self.conv2 = OCTConv_BN_ACT(in_channels=in_channels , out_channels=in_channels//8, kernel_size=1,padding= 0)
            self.conv = OCTConv_BN_ACT(in_channels=in_channels//8, out_channels=out_channels, kernel_size=1, alpha_out=0,padding= 0)

        elif alpha_in == 0:
            self.conv = OCTConv_BN_ACT(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, alpha_in=0,padding= 0)
            self.conv2 = OCTConv_BN_ACT(in_channels=in_channels//8, out_channels=out_channels, kernel_size=1,padding= 0)
        else:
            self.conv = OCTConv_BN_ACT(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1,padding= 0)
            self.conv2 = OCTConv_BN_ACT(in_channels=in_channels//8, out_channels=out_channels, kernel_size=1,padding = 0)

    def forward(self, en_h, en_l, de_h, de_l):

        if self.alpha_in == 0:
            #x = nn.UpsamplingBilinear2d(scale_factor=2)(en_h)
            hh, ll = self.conv(en_h)
            hh, ll = self.conv2((hh,ll))
            hh = nn.UpsamplingBilinear2d(scale_factor=2)(hh)
            ll = nn.UpsamplingBilinear2d(scale_factor=2)(ll)
        elif self.alpha_out == 0:
            h = torch.cat((en_h, de_h), dim=1)
            l = torch.cat((en_l, de_l), dim=1)
            h,l =self.conv2((h,l))
            hh, ll = self.conv((h, l))
            hh = nn.UpsamplingBilinear2d(scale_factor=2)(hh)

        else:

            h = torch.cat((en_h, de_h), dim=1)
            l = torch.cat((en_l, de_l), dim=1)

            hh, ll = self.conv((h, l))
            hh, ll = self.conv2((hh, ll))

            hh = nn.UpsamplingBilinear2d(scale_factor=2)(hh)
            ll = nn.UpsamplingBilinear2d(scale_factor=2)(ll)



        return hh, ll

class decode_all_2(nn.Module):
    def __init__(self, in_channels, out_channels,alpha_in=0.5,alpha_out=0.5):
        super(decode_all_2, self).__init__()
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        if alpha_out==0:
            self.conv2 = OCTConv_BN_ACT(in_channels=in_channels , out_channels=in_channels//8, kernel_size=1,padding= 0)

            self.conv = OCTConv_BN_ACT(in_channels=in_channels//8, out_channels=out_channels, kernel_size=3, alpha_out=0,padding= 1)

        elif alpha_in == 0:
            self.conv = OCTConv_BN_ACT(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, alpha_in=0,padding= 0)
            self.conv2 = OCTConv_BN_ACT(in_channels=in_channels//8, out_channels=out_channels, kernel_size=3,padding= 1)
        else:
            self.conv = OCTConv_BN_ACT(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1,padding= 0)
            self.conv2 = OCTConv_BN_ACT(in_channels=in_channels//8, out_channels=out_channels, kernel_size=3,padding = 1)

    def forward(self, en_h, en_l, de_h, de_l):

        if self.alpha_in == 0:
            #x = nn.UpsamplingBilinear2d(scale_factor=2)(en_h)
            hh, ll = self.conv(en_h)
            hh = nn.UpsamplingBilinear2d(scale_factor=2)(hh)
            ll = nn.UpsamplingBilinear2d(scale_factor=2)(ll)
            hh, ll = self.conv2((hh,ll))

        elif self.alpha_out == 0:
            h = torch.cat((en_h, de_h), dim=1)
            l = torch.cat((en_l, de_l), dim=1)
            hh,ll =self.conv2((h,l))
            hh = nn.UpsamplingBilinear2d(scale_factor=2)(hh)
            ll = nn.UpsamplingBilinear2d(scale_factor=2)(ll)
            hh, ll = self.conv((hh, ll))


        else:

            h = torch.cat((en_h, de_h), dim=1)
            l = torch.cat((en_l, de_l), dim=1)

            hh, ll = self.conv((h, l))
            hh = nn.UpsamplingBilinear2d(scale_factor=2)(hh)
            ll = nn.UpsamplingBilinear2d(scale_factor=2)(ll)
            hh, ll = self.conv2((hh, ll))

        return hh, ll

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

class decode_all_3(nn.Module):
    def __init__(self, in_channels, out_channels,alpha_in=0.5,alpha_out=0.5):
        super(decode_all_3, self).__init__()
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        if alpha_out==0:
            self.conv2 = OCTConv_BN_ACT(in_channels=in_channels , out_channels=in_channels//8, kernel_size=1,padding= 0)
            self.up1 =TransposeX2(in_channels=in_channels//16, out_channels=in_channels//16)
            self.up2 = TransposeX2(in_channels=in_channels // 16, out_channels=in_channels // 16)
            self.conv = OCTConv_BN_ACT(in_channels=in_channels//8, out_channels=out_channels, kernel_size=1, alpha_out=0,padding= 0)

        elif alpha_in == 0:
            self.conv = OCTConv_BN_ACT(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, alpha_in=0,padding= 0)
            self.up1 = TransposeX2(in_channels=in_channels // 16, out_channels=in_channels // 16)
            self.up2 = TransposeX2(in_channels=in_channels // 16, out_channels=in_channels // 16)
            self.conv2 = OCTConv_BN_ACT(in_channels=in_channels//8, out_channels=out_channels, kernel_size=1,padding= 0)
        else:
            self.conv = OCTConv_BN_ACT(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1,padding= 0)
            self.up1 = TransposeX2(in_channels=in_channels // 16, out_channels=in_channels // 16)
            self.up2 = TransposeX2(in_channels=in_channels // 16, out_channels=in_channels // 16)
            self.conv2 = OCTConv_BN_ACT(in_channels=in_channels//8, out_channels=out_channels, kernel_size=1,padding = 0)

    def forward(self, en_h, en_l, de_h, de_l):

        if self.alpha_in == 0:
            #x = nn.UpsamplingBilinear2d(scale_factor=2)(en_h)
            hh, ll = self.conv(en_h)
            hh = self.up1(hh)
            ll = self.up2(ll)
            #hh = nn.UpsamplingBilinear2d(scale_factor=2)(hh)
            #ll = nn.UpsamplingBilinear2d(scale_factor=2)(ll)
            hh, ll = self.conv2((hh,ll))


        elif self.alpha_out == 0:
            h = torch.cat((en_h, de_h), dim=1)
            l = torch.cat((en_l, de_l), dim=1)
            hh,ll =self.conv2((h,l))
            hh = self.up1(hh)
            ll = self.up2(ll)
            #hh = nn.UpsamplingBilinear2d(scale_factor=2)(hh)
            #ll = nn.UpsamplingBilinear2d(scale_factor=2)(ll)
            hh, ll = self.conv((hh, ll))


        else:

            h = torch.cat((en_h, de_h), dim=1)
            l = torch.cat((en_l, de_l), dim=1)

            hh, ll = self.conv((h, l))
            hh = self.up1(hh)
            ll = self.up2(ll)
            #hh = nn.UpsamplingBilinear2d(scale_factor=2)(hh)
            #ll = nn.UpsamplingBilinear2d(scale_factor=2)(ll)
            hh, ll = self.conv2((hh, ll))





        return hh, ll

class OctResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(OctResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, alpha_in=0)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, alpha_out=0, output=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv_BN(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, alpha_in=alpha_in, alpha_out=alpha_out)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, alpha_in, alpha_out, norm_layer, output))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer,
                                alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5, output=output))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_h, x_l = self.layer1(x)
        x_h, x_l = self.layer2((x_h,x_l))
        x_h, x_l = self.layer3((x_h,x_l))
        x_h, x_l = self.layer4((x_h,x_l))
        x = self.avgpool(x_h)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x





class OctResNet_2(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(OctResNet_2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        model = oct_resnet50(pretrained=True)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        #self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        #self.bn1 = norm_layer(self.inplanes)
        #self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.multi1 = multi_Freq(in_channels=128)
        self.multi2 = multi_Freq(in_channels=256)
        self.multi3 = multi_Freq(in_channels=512)
        # Decode

        self.decode_4 = decode_all_3(in_channels=2048, out_channels=512,alpha_in=0)

        self.decode_3 = decode_all_3(in_channels=1024+512, out_channels=256)
        self.decode_2 = decode_all_3(in_channels=512+256, out_channels=128)
        self.decode_1 = decode_all_3(in_channels=256+128, out_channels=64,alpha_out=0)

        self.decode_0 = decode_00(in_channels=128, out_channels=64)

        self.seg = nn.Conv2d(64, num_classes, kernel_size=1)
        """
        self.seg =nn.Sequential(
            nn.Conv2d(64, 32, 1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, padding=0), 
        )"""

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv_BN(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, alpha_in=alpha_in,
                        alpha_out=alpha_out)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, alpha_in, alpha_out, norm_layer, output))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer,
                                alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5, output=output))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x_h = self.relu(x)  # 1*64*128*128

        x_l = self.maxpool(x_h)  # 1*64*64*64

        # x_h, x_l = self.layer1(x) # x_h: torch.Size([1, 128, 64, 64]) x_l: torch.Size([1, 128, 32, 32])
        en_1_h, en_1_l = self.layer1(x_l)
        # x_h, x_l = self.layer2((x_h,x_l)) # x_h: torch.Size([1, 256, 32, 32]) x_l: torch.Size([1, 256, 16, 16])
        en_2_h, en_2_l = self.layer2((en_1_h, en_1_l))
        # x_h, x_l = self.layer3((x_h,x_l)) # x_h: torch.Size([1, 512, 16, 16]) x_l: torch.Size([1, 512, 8, 8])
        en_3_h, en_3_l = self.layer3((en_2_h, en_2_l))
        # x_h, x_l = self.layer4((x_h,x_l)) # x_h: torch.Size([1, 2048, 8, 8]) x_l为空，没有任何值
        en_4_h, _ = self.layer4((en_3_h, en_3_l))

        en_1_h, en_1_l = self.multi1(en_1_h, en_1_l)
        en_2_h, en_2_l = self.multi2(en_2_h, en_2_l)
        en_3_h, en_3_l = self.multi3(en_3_h, en_3_l)

        # 解码部分

        de_4_h, de_4_l = self.decode_4(en_4_h,_,_,_)  # 1,512,16,16    1,512,8,8
        de_3_h, de_3_l = self.decode_3(en_3_h, en_3_l, de_4_h,
                                       de_4_l)  # torch.Size([1, 256, 32, 32]) torch.Size([1, 256, 16, 16])
        de_2_h, de_2_l = self.decode_2(en_2_h, en_2_l, de_3_h,
                                       de_3_l)  # torch.Size([1, 128, 64, 64]) torch.Size([1, 128, 32, 32])
        de_1_h, _ = self.decode_1(en_1_h, en_1_l, de_2_h, de_2_l)  # torch.Size([1, 64, 128, 128])

        de_0_h = self.decode_0(de_1_h, x_h)  # torch.Size([1, 64, 256, 256])

        out = self.seg(de_0_h)

        """
        x = self.avgpool(x_h)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        """
        return out





class OctResNet_multi_freq(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(OctResNet_multi_freq, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, alpha_in=0)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, alpha_out=0, output=True)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # muti_freq

        self.multi1 = multi_Freq(in_channels=128)
        self.multi2 = multi_Freq(in_channels=256)
        self.multi3 = multi_Freq(in_channels=512)

        # Decode

        self.decode_4 = decode_4(in_channels=2048, out_channels=1024)
        self.decode_3 = decode(in_channels=2048, out_channels=512)
        self.decode_2 = decode(in_channels=1024, out_channels=256)
        self.decode_1 = decode_1(in_channels=512, out_channels=64)

        self.decode_0 = decode_00(in_channels=128, out_channels=64)

        self.seg = nn.Conv2d(64, num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv_BN(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, alpha_in=alpha_in,
                        alpha_out=alpha_out)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, alpha_in, alpha_out, norm_layer, output))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer,
                                alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5, output=output))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x_h = self.relu(x)  # 1*64*128*128

        x_l = self.maxpool(x_h)  # 1*64*64*64

        # x_h, x_l = self.layer1(x) # x_h: torch.Size([1, 128, 64, 64]) x_l: torch.Size([1, 128, 32, 32])
        en_1_h, en_1_l = self.layer1(x_l)
        # en_1_h, en_1_l = self.multi1(en_1_h, en_1_l)
        # x_h, x_l = self.layer2((x_h,x_l)) # x_h: torch.Size([1, 256, 32, 32]) x_l: torch.Size([1, 256, 16, 16])
        en_2_h, en_2_l = self.layer2((en_1_h, en_1_l))
        # en_2_h, en_2_l = self.multi2(en_2_h, en_2_l)
        # x_h, x_l = self.layer3((x_h,x_l)) # x_h: torch.Size([1, 512, 16, 16]) x_l: torch.Size([1, 512, 8, 8])
        en_3_h, en_3_l = self.layer3((en_2_h, en_2_l))
        # en_3_h, en_3_l = self.multi3(en_3_h, en_3_l)
        # x_h, x_l = self.layer4((x_h,x_l)) # x_h: torch.Size([1, 2048, 8, 8]) x_l为空，没有任何值
        en_4_h, _ = self.layer4((en_3_h, en_3_l))

        en_1_h, en_1_l = self.multi1(en_1_h, en_1_l)
        en_2_h, en_2_l = self.multi2(en_2_h, en_2_l)
        en_3_h, en_3_l = self.multi3(en_3_h, en_3_l)

        # 解码部分

        de_4_h, de_4_l = self.decode_4(en_4_h)  #
        de_3_h, de_3_l = self.decode_3(en_3_h, en_3_l, de_4_h,
                                       de_4_l)  # torch.Size([1, 256, 32, 32]) torch.Size([1, 256, 16, 16])
        de_2_h, de_2_l = self.decode_2(en_2_h, en_2_l, de_3_h,
                                       de_3_l)  # torch.Size([1, 128, 64, 64]) torch.Size([1, 128, 32, 32])
        de_1_h, _ = self.decode_1(en_1_h, en_1_l, de_2_h, de_2_l)  # torch.Size([1, 64, 128, 128])

        de_0_h = self.decode_0(de_1_h, x_h)  # torch.Size([1, 64, 256, 256])

        out = self.seg(de_0_h)

        """
        x = self.avgpool(x_h)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        """
        return out

###################  oct+MFtransformer+multi_scale   ######################################################

class OctResNet_3(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(OctResNet_3, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        model = oct_resnet50(pretrained=False)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        #self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        #self.bn1 = norm_layer(self.inplanes)
        #self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.multi1 = multi_Freq(in_channels=128)
        self.multi2 = multi_Freq(in_channels=256)
        self.multi3 = multi_Freq(in_channels=512)

        # multi_scale
        self.multi_scale_fusion = multiscale(in_channels=256,out_channels=256,fin_channels=1)

        # Decode

        self.decode_4 = decode_all_3(in_channels=2048, out_channels=512,alpha_in=0)

        self.decode_3 = decode_all_3(in_channels=1024+512, out_channels=256)
        self.decode_2 = decode_all_3(in_channels=512+256, out_channels=128)
        self.decode_1 = decode_all_3(in_channels=256+128, out_channels=64,alpha_out=0)

        self.decode_0 = decode_00(in_channels=128, out_channels=64)

        self.seg = nn.Conv2d(64, num_classes, kernel_size=1)
        self.out = nn.Conv2d(2,1,1)

        self.seg1 = nn.Conv2d(128, num_classes, kernel_size=1)
        """
        self.seg =nn.Sequential(
            nn.Conv2d(64, 32, 1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, padding=0), 
        )"""

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv_BN(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, alpha_in=alpha_in,
                        alpha_out=alpha_out)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, alpha_in, alpha_out, norm_layer, output))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer,
                                alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5, output=output))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x_h = self.relu(x)  # 1*64*128*128

        x_l = self.maxpool(x_h)  # 1*64*64*64

        # x_h, x_l = self.layer1(x) # x_h: torch.Size([1, 128, 64, 64]) x_l: torch.Size([1, 128, 32, 32])
        en_1_h, en_1_l = self.layer1(x_l)
        # x_h, x_l = self.layer2((x_h,x_l)) # x_h: torch.Size([1, 256, 32, 32]) x_l: torch.Size([1, 256, 16, 16])
        en_2_h, en_2_l = self.layer2((en_1_h, en_1_l))
        # x_h, x_l = self.layer3((x_h,x_l)) # x_h: torch.Size([1, 512, 16, 16]) x_l: torch.Size([1, 512, 8, 8])
        en_3_h, en_3_l = self.layer3((en_2_h, en_2_l))
        # x_h, x_l = self.layer4((x_h,x_l)) # x_h: torch.Size([1, 2048, 8, 8]) x_l为空，没有任何值
        en_4_h, _ = self.layer4((en_3_h, en_3_l))

        en_1_h, en_1_l = self.multi1(en_1_h, en_1_l)
        en_2_h, en_2_l = self.multi2(en_2_h, en_2_l)
        en_3_h, en_3_l = self.multi3(en_3_h, en_3_l)

        # 解码部分

        de_4_h, de_4_l = self.decode_4(en_4_h,_,_,_)  # 1,512,16,16    1,512,8,8
        de_3_h, de_3_l = self.decode_3(en_3_h, en_3_l, de_4_h,
                                       de_4_l)  # torch.Size([1, 256, 32, 32]) torch.Size([1, 256, 16, 16])
        de_2_h, de_2_l = self.decode_2(en_2_h, en_2_l, de_3_h,
                                       de_3_l)  # torch.Size([1, 128, 64, 64]) torch.Size([1, 128, 32, 32])
        de_1_h, _ = self.decode_1(en_1_h, en_1_l, de_2_h, de_2_l)  # torch.Size([1, 64, 128, 128])

        de_0_h = self.decode_0(de_1_h, x_h)  # torch.Size([1, 64, 256, 256])

        multiscale_fusion = self.multi_scale_fusion(en_1_h,en_2_h,en_3_h)

        out = torch.cat((de_0_h,multiscale_fusion),dim=1)
        out = self.seg1(out)



        #out = self.seg(de_0_h)
        #out = torch.cat((out,multiscale_fusion),dim=1)

        #out = self.out(out)

        """
        x = self.avgpool(x_h)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        """
        return out

#################    oct+multi_scale             #########################
class OctResNet_4(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(OctResNet_4, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        model = oct_resnet50(pretrained=True)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        #self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        #self.bn1 = norm_layer(self.inplanes)
        #self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        #self.multi1 = multi_Freq(in_channels=128)
        #self.multi2 = multi_Freq(in_channels=256)
        #self.multi3 = multi_Freq(in_channels=512)

        # multi_scale
        self.multi_scale_fusion = multiscale(in_channels=256,out_channels=256,fin_channels=1)

        # Decode

        self.decode_4 = decode_all_3(in_channels=2048, out_channels=512,alpha_in=0)

        self.decode_3 = decode_all_3(in_channels=1024+512, out_channels=256)
        self.decode_2 = decode_all_3(in_channels=512+256, out_channels=128)
        self.decode_1 = decode_all_3(in_channels=256+128, out_channels=64,alpha_out=0)

        self.decode_0 = decode_00(in_channels=128, out_channels=64)

        self.seg = nn.Conv2d(64, num_classes, kernel_size=1)
        self.out = nn.Conv2d(2,1,1)

        self.seg1 = nn.Conv2d(128, num_classes, kernel_size=1)
        """
        self.seg =nn.Sequential(
            nn.Conv2d(64, 32, 1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, padding=0), 
        )"""

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv_BN(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, alpha_in=alpha_in,
                        alpha_out=alpha_out)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, alpha_in, alpha_out, norm_layer, output))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer,
                                alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5, output=output))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x_h = self.relu(x)  # 1*64*128*128

        x_l = self.maxpool(x_h)  # 1*64*64*64

        # x_h, x_l = self.layer1(x) # x_h: torch.Size([1, 128, 64, 64]) x_l: torch.Size([1, 128, 32, 32])
        en_1_h, en_1_l = self.layer1(x_l)
        # x_h, x_l = self.layer2((x_h,x_l)) # x_h: torch.Size([1, 256, 32, 32]) x_l: torch.Size([1, 256, 16, 16])
        en_2_h, en_2_l = self.layer2((en_1_h, en_1_l))
        # x_h, x_l = self.layer3((x_h,x_l)) # x_h: torch.Size([1, 512, 16, 16]) x_l: torch.Size([1, 512, 8, 8])
        en_3_h, en_3_l = self.layer3((en_2_h, en_2_l))
        # x_h, x_l = self.layer4((x_h,x_l)) # x_h: torch.Size([1, 2048, 8, 8]) x_l为空，没有任何值
        en_4_h, _ = self.layer4((en_3_h, en_3_l))

        #en_1_h, en_1_l = self.multi1(en_1_h, en_1_l)
        #en_2_h, en_2_l = self.multi2(en_2_h, en_2_l)
        #en_3_h, en_3_l = self.multi3(en_3_h, en_3_l)

        # 解码部分

        de_4_h, de_4_l = self.decode_4(en_4_h,_,_,_)  # 1,512,16,16    1,512,8,8
        de_3_h, de_3_l = self.decode_3(en_3_h, en_3_l, de_4_h,
                                       de_4_l)  # torch.Size([1, 256, 32, 32]) torch.Size([1, 256, 16, 16])
        de_2_h, de_2_l = self.decode_2(en_2_h, en_2_l, de_3_h,
                                       de_3_l)  # torch.Size([1, 128, 64, 64]) torch.Size([1, 128, 32, 32])
        de_1_h, _ = self.decode_1(en_1_h, en_1_l, de_2_h, de_2_l)  # torch.Size([1, 64, 128, 128])

        de_0_h = self.decode_0(de_1_h, x_h)  # torch.Size([1, 64, 256, 256])

        multiscale_fusion = self.multi_scale_fusion(en_1_h,en_2_h,en_3_h)

        out = torch.cat((de_0_h,multiscale_fusion),dim=1)
        out = self.seg1(out)



        #out = self.seg(de_0_h)
        #out = torch.cat((out,multiscale_fusion),dim=1)

        #out = self.out(out)

        """
        x = self.avgpool(x_h)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        """
        return out

def oct_resnet26(pretrained=False, **kwargs):
    """Constructs a Octave ResNet-26 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OctResNet(Bottleneck, [2, 2, 2, 2], **kwargs)
    return model


def oct_resnet50(pretrained=False, **kwargs):
    """Constructs a Octave ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = OctResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained == True:
        # model = nn.DataParallel(model)
        state_dict = torch.load('oct_resnet50_cosine.pth')
        model.load_state_dict(state_dict, strict=False)
    return model


##################    使用这个       ########################################


def oct_resnet50_2(pretrained=False, **kwargs):
    """Constructs a Octave ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = OctResNet_2(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained == True:
        # model = nn.DataParallel(model)
        state_dict = torch.load('mymodel/Oct_ResNet/oct_resnet50_cosine.pth')
        model.load_state_dict(state_dict, strict=False)
    return model


def oct_resnet50_multifreq(pretrained=False, **kwargs):
    """Constructs a Octave ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = OctResNet_multi_freq(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained == True:
        # model = nn.DataParallel(model)
        state_dict = torch.load('mymodel/Oct_ResNet/oct_resnet50_cosine.pth')
        model.load_state_dict(state_dict, strict=False)
    return model


def oct_resnet50_fusion(pretrained=False, **kwargs):
    """Constructs a Octave ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = OctResNet_3(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained == True:
        # model = nn.DataParallel(model)
        state_dict = torch.load('oct_resnet50_cosine.pth')
        model.load_state_dict(state_dict, strict=False)
    return model

def oct_resnet50_nofreq_fusion(pretrained=False, **kwargs):
    """Constructs a Octave ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = OctResNet_4(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained == True:
        # model = nn.DataParallel(model)
        state_dict = torch.load('mymodel/Oct_ResNet/oct_resnet50_cosine.pth')
        model.load_state_dict(state_dict, strict=False)
    return model



####################################################################################################


if __name__ == '__main__':
    import os
    from PIL import Image
    from torchvision import transforms

    # model = IRPB(in_ch=64, out_ch=64, branch=4, rate=(1, 2, 4, 5))
    model = oct_resnet50_fusion(pretrained=True)
    # print(model)

    model.eval()
    x = torch.randn(1, 3, 256, 256)
    print(model)
    y_h = model(x)
    print(y_h.size())
    """
    from thop import profile

    flops, params = profile(model, inputs=(x,))
    print('GFLOPS:{:.4f}||Mparams:{:.4f}'.format(flops / 1e9, params / 1e6))
    """
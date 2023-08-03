import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3)
        return scale



class FEM(nn.Module):
    def __init__(self, inplanes, channel_rate=2, reduction_ratio=16):
        super(FEM, self).__init__()

        self.in_channels = inplanes
        self.inter_channels = inplanes // channel_rate
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.common_v = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                         padding=0)

        self.Trans_s = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        nn.init.constant_(self.Trans_s[1].weight, 0)
        nn.init.constant_(self.Trans_s[1].bias, 0)

        self.Trans_q = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        nn.init.constant_(self.Trans_q[1].weight, 0)
        nn.init.constant_(self.Trans_q[1].bias, 0)

        #
        self.key = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)
        self.query = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)

        self.dropout = nn.Dropout(0.1)
        self.ChannelGate = ChannelGate(self.in_channels, pool_types=['avg'], reduction_ratio=reduction_ratio)

    def forward(self, q, s):
        batch_size, channels, height_q, width_q = q.shape
        batch_size, channels, height_s, width_s = s.shape

        # Cross-image information communication

        # common feature learning
        v_q = self.common_v(q).view(batch_size, self.inter_channels, -1)
        v_q = v_q.permute(0, 2, 1) #(B,HW,C)

        v_s = self.common_v(s).view(batch_size, self.inter_channels, -1)
        v_s = v_s.permute(0, 2, 1) #(B,HW,C)

        k_x = self.key(s).view(batch_size, self.inter_channels, -1)
        k_x = k_x.permute(0, 2, 1)

        q_x = self.query(q).view(batch_size, self.inter_channels, -1)

        A_s = torch.matmul(k_x, q_x)
        attention_s = F.softmax(A_s, dim=-1)

        A_q = A_s.permute(0, 2, 1).contiguous()
        attention_q = F.softmax(A_q, dim=-1)

        p_s = torch.matmul(attention_s, v_s)
        p_s = p_s.permute(0, 2, 1).contiguous()
        p_s = p_s.view(batch_size, self.inter_channels, height_s, width_s)
        # individual feature learning for s
        p_s = self.Trans_s(p_s)
        # Intra-image channel attention
        E_s = self.ChannelGate(s) * p_s
        E_s = E_s + s

        q_s = torch.matmul(attention_q, v_q)
        q_s = q_s.permute(0, 2, 1).contiguous()
        q_s = q_s.view(batch_size, self.inter_channels, height_q, width_q)
        # individual feature learning for q
        q_s = self.Trans_q(q_s)
        # Intra-image channel attention
        E_q = self.ChannelGate(q) * q_s
        E_q = E_q + q

        return E_q, E_s


class mutiFreq(nn.Module):
    def __init__(self,in_channels):
        super(mutiFreq, self).__init__()

        self.in_channels = in_channels
        self.to_qkv_h = depthwise_separable_conv(in_ch=in_channels,out_ch=in_channels*2)
        self.to_qkv_l = nn.Conv2d(in_channels=in_channels,out_channels=in_channels*2,kernel_size=1,stride=1,padding=0)
        self.out_h = depthwise_separable_conv(in_ch=in_channels,out_ch=in_channels)
        self.out_l = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,padding=0)



        #####初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self,high_f,low_f):
        # high_f = torch.randn(1,256,16,16) low_f = torch.randn(1,256,8,8)
        b,c,h,w = high_f.shape
        qkv_h = self.to_qkv_h(high_f) #(1,512,16,16)
        q_h,v_h = qkv_h.chunk(2,dim=1) #(1,256,16,16)
        qkv_l = self.to_qkv_l(low_f)
        k_l,v_l = qkv_l.chunk(2,dim=1)

        q_h = rearrange(q_h, 'b c h w -> b c (h w)', b=b,c=c,h=h,w=w)
        v_h = rearrange(v_h, 'b c h w -> b c (h w)', b=b, c=c, h=h, w=w)
        k_l = rearrange(k_l, 'b c h w -> b c (h w)', b=b,c=c,h=h//2,w=w//2)
        v_l = rearrange(v_l, 'b c h w -> b c (h w)', b=b, c=c, h=h // 2, w=w // 2)

        q_k_attn = torch.einsum('bcq,bck->bqk', q_h, k_l) # b,HW,hw
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        scale = c ** (-0.5)
        q_k_attn = q_k_attn*scale

        low_f_out = torch.einsum('bHh,bcH->bch',q_k_attn,v_h)
        low_f_out = low_f_out.contiguous().view(b, c, h//2 , w//2)
        low_f_out = self.out_l(low_f_out)

        high_f_out = torch.einsum('bHh,bch->bcH', q_k_attn, v_l)
        high_f_out = high_f_out.contiguous().view(b, c, h, w)
        high_f_out = self.out_h(high_f_out)



        return high_f_out,low_f_out

class depthwise_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias, stride=stride)
        #self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        #out = self.pointwise(out)

        return out

class mutiFreq_heads(nn.Module):
    def __init__(self,in_channels,num_heads=8):
        super(mutiFreq_heads, self).__init__()

        self.num_heads = num_heads
        head_dim = in_channels // num_heads
        self.scale = head_dim ** -0.5

        self.in_channels = in_channels
        self.to_qkv_h = depthwise_separable_conv(in_ch=in_channels,out_ch=in_channels*2)
        self.to_qkv_l = nn.Conv2d(in_channels=in_channels,out_channels=in_channels*2,kernel_size=1,stride=1,padding=0)
        self.out_h = depthwise_separable_conv(in_ch=in_channels,out_ch=in_channels)
        self.out_l = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,padding=0)

        #####初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self,high_f,low_f):
        # high_f = torch.randn(1,256,16,16) low_f = torch.randn(1,256,8,8)
        b,c,h,w = high_f.shape
        qkv_h = self.to_qkv_h(high_f) #(1,512,16,16)
        q_h,v_h = qkv_h.chunk(2,dim=1) #(1,256,16,16)
        q_h = q_h.reshape(b,self.num_heads,c//self.num_heads,h,w)
        v_h = v_h.reshape(q_h.shape)
        qkv_l = self.to_qkv_l(low_f)
        k_l,v_l = qkv_l.chunk(2,dim=1)
        k_l = k_l.reshape(b,self.num_heads,c//self.num_heads,h//2,w//2)
        v_l = v_l.reshape(k_l.shape)

        q_h = rearrange(q_h, 'b nh c h w -> b nh c (h w)', b=b,nh=self.num_heads, c=c//self.num_heads, h=h,w=w)
        v_h = rearrange(v_h, 'b nh c h w -> b nh c (h w)', b=b,nh=self.num_heads, c=c//self.num_heads, h=h, w=w)
        k_l = rearrange(k_l, 'b nh c h w -> b nh c (h w)', b=b,nh=self.num_heads, c=c//self.num_heads, h=h//2,w=w//2)
        v_l = rearrange(v_l, 'b nh c h w -> b nh c (h w)', b=b,nh=self.num_heads, c=c//self.num_heads, h=h // 2, w=w // 2)

        q_k_attn = torch.einsum('bncH,bnch->bnHh', q_h, k_l) # b,HW,hw
        q_k_attn = q_k_attn*self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)


        low_f_out = torch.einsum('bnHh,bncH->bnch',q_k_attn,v_h)
        low_f_out = low_f_out.contiguous()
        low_f_out = rearrange(low_f_out, 'b nh c (h w) -> b (nh c) h w', b=b, nh=self.num_heads, c=c // self.num_heads, h=h//2, w=w//2)
        low_f_out = self.out_l(low_f_out)

        high_f_out = torch.einsum('bnHh,bnch->bncH', q_k_attn, v_l)
        high_f_out = high_f_out.contiguous()
        high_f_out = rearrange(high_f_out, 'b nh c (h w) -> b (nh c) h w', b=b, nh=self.num_heads, c=c // self.num_heads, h=h, w=w)
        high_f_out = self.out_h(high_f_out)



        return high_f_out,low_f_out



class rel_mutiFreq_heads(nn.Module):
    def __init__(self,in_channels,num_heads=8):
        super(rel_mutiFreq_heads, self).__init__()

        self.num_heads = num_heads
        head_dim = in_channels // num_heads
        self.scale = head_dim ** -0.5

        self.in_channels = in_channels
        self.to_qkv_h = depthwise_separable_conv(in_ch=in_channels,out_ch=in_channels*2)
        self.to_qkv_l = nn.Conv2d(in_channels=in_channels,out_channels=in_channels*2,kernel_size=1,stride=1,padding=0)
        self.out_h = depthwise_separable_conv(in_ch=in_channels,out_ch=in_channels)
        self.out_l = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,padding=0)

        #####初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self,high_f,low_f):
        # high_f = torch.randn(1,256,16,16) low_f = torch.randn(1,256,8,8)
        b,c,h,w = high_f.shape
        qkv_h = self.to_qkv_h(high_f) #(1,512,16,16)
        q_h,v_h = qkv_h.chunk(2,dim=1) #(1,256,16,16)
        q_h = q_h.reshape(b,self.num_heads,c//self.num_heads,h,w)
        v_h = v_h.reshape(q_h.shape)
        qkv_l = self.to_qkv_l(low_f)
        k_l,v_l = qkv_l.chunk(2,dim=1)
        k_l = k_l.reshape(b,self.num_heads,c//self.num_heads,h//2,w//2)
        v_l = v_l.reshape(k_l.shape)

        q_h = rearrange(q_h, 'b nh c h w -> b nh c (h w)', b=b,nh=self.num_heads, c=c//self.num_heads, h=h,w=w)
        v_h = rearrange(v_h, 'b nh c h w -> b nh c (h w)', b=b,nh=self.num_heads, c=c//self.num_heads, h=h, w=w)
        k_l = rearrange(k_l, 'b nh c h w -> b nh c (h w)', b=b,nh=self.num_heads, c=c//self.num_heads, h=h//2,w=w//2)
        v_l = rearrange(v_l, 'b nh c h w -> b nh c (h w)', b=b,nh=self.num_heads, c=c//self.num_heads, h=h // 2, w=w // 2)

        q_k_attn = torch.einsum('bncH,bnch->bnHh', q_h, k_l) # b,HW,hw
        q_k_attn = q_k_attn*self.scale



        q_k_attn = F.softmax(q_k_attn, dim=-1)


        low_f_out = torch.einsum('bnHh,bncH->bnch',q_k_attn,v_h)
        low_f_out = low_f_out.contiguous()
        low_f_out = rearrange(low_f_out, 'b nh c (h w) -> b (nh c) h w', b=b, nh=self.num_heads, c=c // self.num_heads, h=h//2, w=w//2)
        low_f_out = self.out_l(low_f_out)

        high_f_out = torch.einsum('bnHh,bnch->bncH', q_k_attn, v_l)
        high_f_out = high_f_out.contiguous()
        high_f_out = rearrange(high_f_out, 'b nh c (h w) -> b (nh c) h w', b=b, nh=self.num_heads, c=c // self.num_heads, h=h, w=w)
        high_f_out = self.out_h(high_f_out)



        return high_f_out,low_f_out

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class double_multiFreq(nn.Module):
    def __init__(self, in_channels,):
        super(double_multiFreq, self).__init__()

        norm_layer = nn.BatchNorm2d
        self.conv_down = conv1x1(in_channels, in_channels)
        self.bn1 = norm_layer(in_channels)
        self.conv_down_2 = conv1x1(in_channels, in_channels)
        self.bn1_2 = norm_layer(in_channels)

        self.att1 = mutiFreq_heads(in_channels=in_channels)
        self.att2 = mutiFreq_heads(in_channels=in_channels)
        self.conv_up = conv1x1(in_channels, in_channels)
        self.bn2 = norm_layer(in_channels)

        self.conv_up_2 = conv1x1(in_channels, in_channels)
        self.bn2_2 = norm_layer(in_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_f, low_f):
        identity_h, identity_l = high_f, low_f

        high_f = self.conv_down(high_f)
        high_f = self.bn1(high_f)
        high_f = self.relu(high_f)

        low_f = self.conv_down_2(low_f)
        low_f = self.bn1_2(low_f)
        low_f = self.relu(low_f)

        high,low = self.att1(high_f,low_f)
        #high,low = self.att2(high,low)

        high = self.relu(high)
        low = self.relu(low)

        high = high+identity_h
        low = low+identity_l
        #high = self.conv_up(high)
        high = self.bn2(high)

        #low = self.conv_up_2(low)
        low = self.bn2_2(low)



        high = self.relu(high)
        low = self.relu(low)

        return high, low

class one_multiFreq(nn.Module):
    def __init__(self, in_channels,):
        super(one_multiFreq, self).__init__()

        norm_layer = nn.BatchNorm2d
        self.conv_down = conv1x1(in_channels, in_channels)
        self.bn1 = norm_layer(in_channels)
        self.conv_down_2 = conv1x1(in_channels, in_channels)
        self.bn1_2 = norm_layer(in_channels)

        self.att1 = mutiFreq_heads(in_channels=in_channels)
        #self.att2 = mutiFreq_heads(in_channels=in_channels)
        self.conv_up = conv1x1(in_channels, in_channels)
        self.bn2 = norm_layer(in_channels)

        self.conv_up_2 = conv1x1(in_channels, in_channels)
        self.bn2_2 = norm_layer(in_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_f, low_f):
        identity_h, identity_l = high_f, low_f

        high_f = self.conv_down(high_f)
        high_f = self.bn1(high_f)
        high_f = self.relu(high_f)

        low_f = self.conv_down_2(low_f)
        low_f = self.bn1_2(low_f)
        low_f = self.relu(low_f)

        high, low = self.att1(high_f, low_f)
        # high,low = self.att1(high,low)

        high = self.relu(high)
        low = self.relu(low)

        high = high + identity_h
        low = low + identity_l
        # high = self.conv_up(high)
        high = self.bn2(high)

        # low = self.conv_up_2(low)
        low = self.bn2_2(low)

        high = self.relu(high)
        low = self.relu(low)
        """
        identity_h, identity_l = high_f, low_f

        high_f = self.conv_down(high_f)
        high_f = self.bn1(high_f)
        high_f = self.relu(high_f)

        low_f = self.conv_down_2(low_f)
        low_f = self.bn1_2(low_f)
        low_f = self.relu(low_f)

        high,low = self.att1(high_f,low_f)
        #high,low = self.att1(high,low)

        high = self.relu(high)
        low = self.relu(low)

        high = high + identity_h
        low = low + identity_l
        # high = self.conv_up(high)
        high = self.bn2(high)

        # low = self.conv_up_2(low)
        low = self.bn2_2(low)

        high = self.relu(high)
        low = self.relu(low)
        """
        return high, low
class multi_Freq(nn.Module):
    def __init__(self,in_channels):
        super(multi_Freq, self).__init__()

        self.transformer1 = one_multiFreq(in_channels=in_channels)
        self.transformer2 = one_multiFreq(in_channels=in_channels)
        #self.transformer3 = one_multiFreq(in_channels=in_channels)
    def forward(self,high,low):

        high, low = self.transformer1(high, low)
        high, low = self.transformer2(high, low)
        #high, low = self.transformer3(high, low)
        return high,low


class freqtransformer(nn.Module):
    def __init__(self,in_channels,out_channels,num_block=2,name_block=one_multiFreq,dowmsample=True):
        super(freqtransformer, self).__init__()
        self.in_channels = in_channels
        self.dowmsample = None
        if dowmsample == True:
            self.dowmsample = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=2)
        else:
            out_channels = in_channels
        self.layer = self.make_layer(in_channels=out_channels,num_block=num_block,name_block=name_block)

    def make_layer(self,in_channels, num_block, name_block=one_multiFreq):
        layer = []
        for i in range(0,num_block):
            layer.append(name_block(in_channels=in_channels))
        return layer

    def forward(self,high,low):

        if self.dowmsample != None:
            high = self.dowmsample(high)
            low = self.dowmsample(low)
        for m in self.layer:
            high,low = m(high, low)

        return high,low

class bottomtransformer(nn.Module):
    def __init__(self,in_channels,out_channels,num_block = 2):
        super(bottomtransformer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=2)
        self.conv2 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1)
        self.layer = self.make_layer(in_channels=out_channels,num_block=num_block)
    def make_layer(self,in_channels, num_block=2):
        layer = []
        for i in range(0,num_block):
            layer.append(one_multiFreq(in_channels=in_channels))
        return layer

    def forward(self,high,low):

        high = self.conv1(high)
        low = self.conv2(low)
        for m in self.layer:
            high, low = m(high, low)

        return high,low








if __name__=='__main__':
    num_heads = [2, 4, 8]

    x = torch.randn(1,256,16,16)
    y = torch.randn(1,256,8,8)
    model = double_multiFreq(in_channels=256)
    z = model(x,y)
    print(z[0].size())
    print(z[1].size())
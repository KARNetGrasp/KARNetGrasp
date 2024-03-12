import torch
import torch.nn as nn

import math 


class crossAttn(nn.Module): 
    def __init__(self, img_dim, word_dim, hidden_dim=None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = img_dim

        self.hidden_dim = hidden_dim
        self.key_dim = word_dim

        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, kdim=word_dim, vdim=word_dim)
        self.IN = nn.InstanceNorm2d(hidden_dim, affine=True)


    def forward(self, img, word, word_mask=None):
        img_residual = img
        B, C, H, W  = img.shape
        vis_pos = self.pos2d(C, H, W)


        img_q_hw = img.view(B, -1, H* W).permute(2, 0, 1)
        word = word.permute(1, 0, 2)
        
        img_q_hw = self.with_pos_embed(img_q_hw, vis_pos)
  
        img_HW_exp = self.cross_attn(img_q_hw, word, word, key_padding_mask=word_mask)[0]
        img_HW_exp = img_HW_exp.permute(1, 2, 0).reshape(B, -1, H, W)
        img_HW = img_residual + img_HW_exp
        img_HW = self.IN(img_HW)

        return img_HW
    
    @staticmethod
    def pos2d(d_model, height, width):
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)

        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)


class selfAttn(nn.Module):
    def __init__(self, img_dim, hidden_dim=None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = img_dim

        self.hidden_dim = hidden_dim

        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)
        self.IN = nn.InstanceNorm2d(hidden_dim, affine=True)


    def forward(self, img):
        img_residual = img
        B, C, H, W  = img.shape
        vis_pos = self.pos2d(C, H, W)


        img_q_hw = img.view(B, -1, H* W).permute(2, 0, 1)
        
        img_q_hw = self.with_pos_embed(img_q_hw, vis_pos)
        img_v_hw = img_residual.view(B, -1, H* W).permute(2 ,0 ,1)
        img_HW_exp = self.self_attn(img_q_hw, img_q_hw, img_v_hw)[0]
        img_HW_exp = img_HW_exp.permute(1, 2, 0).reshape(B, -1, H, W)
        img_HW = img_residual + img_HW_exp
        img_HW = self.IN(img_HW)

        return img_HW
    
    @staticmethod
    def pos2d(d_model, height, width):
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)

        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)


class SeqCrossAttn(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim=None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = query_dim

        self.hidden_dim = hidden_dim
        self.key_dim = key_dim

        self.Linear_q = nn.Linear(query_dim, hidden_dim)
        self.linear_k = nn.Linear(key_dim, hidden_dim)
        self.linear_v = nn.Linear(key_dim, hidden_dim)
        self.LN = nn.LayerNorm(hidden_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, query, key, key_mask=None):
        query_residual = query

        B, L, C = query.shape

        query = self.Linear_q(query)

        key = self.linear_k(key)
        value = self.linear_v(key)


        attn = torch.matmul(query, key.permute(0, 2, 1))

        attn = (self.hidden_dim ** -.5) * attn

        if key_mask is not None:
            attn = attn.masked_fill(key_mask == 0, -1e9)
        
        attn = torch.softmax(attn, dim=-1)

        query = torch.matmul(attn, value)

        query = query_residual + query
        query = self.LN(query)

        return query


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel, ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel, ratio=ratio)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):

        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out



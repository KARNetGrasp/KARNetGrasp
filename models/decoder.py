import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusion import FGOP
  

    
class neck(nn.Module):
    def __init__(self, feats_dims=[2048, 1024, 512, 256], word_dim=512, hidden_dim=128):
        super().__init__()
        hidden_size = hidden_dim
        c4_size = feats_dims[0]
        c3_size = feats_dims[1]
        c2_size = feats_dims[2]
        c1_size = feats_dims[3]

        self.fgop0 = FGOP(in_dim=c4_size, out_dim=hidden_size, word_dim=word_dim)

        self.fgop1 = FGOP(in_dim=hidden_size+c3_size, out_dim=hidden_size, word_dim=word_dim)


        self.fgop2 = FGOP(in_dim=hidden_size + c2_size, out_dim=hidden_size, word_dim=word_dim)

        self.fgop3 = FGOP(in_dim=hidden_size + c1_size, out_dim=hidden_size, word_dim=word_dim)


    def forward(self, img_feats, word, mask=None):
        '''
            img_feats: [B, C1, H/8, W/8], [B, C2, H/16, W/16], [B, C3, H/32, W/32]
            word_feats: [B, C, N]
            sent_feats: [B, C]

        '''
        x_c4, x_c3, x_c2, x_c1 = img_feats

        features = []
        x = self.fgop0(x_c4, word, mask)
        features.append(x)
        # fuse Y4 and Y3
        if x.size(-2) < x_c3.size(-2) or x.size(-1) < x_c3.size(-1):
            x = F.interpolate(input=x, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear', align_corners=True)

        x = torch.cat([x, x_c3], dim=1)

        x = self.fgop1(x, word, mask)
        features.append(x)
        # fuse top-down features and Y2 features
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c2], dim=1)

        x = self.fgop2(x, word, mask)
        features.append(x)
        # fuse top-down features and Y1 features
        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(input=x, size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c1], dim=1)

        x = self.fgop3(x, word, mask)
        features.append(x)

        return features
    
class segmentation(nn.Module):
    def __init__(self, in_dim, hidden_dim, img_size, word_dim, fus=True):
        super().__init__()
        self.img_size = img_size
        self.fus = fus
        self.conv1x1 = nn.Conv2d(4*in_dim, 2*hidden_dim, kernel_size=1)
        if self.fus:
            self.fgop = FGOP(in_dim=2*hidden_dim, out_dim=2*hidden_dim, word_dim=word_dim)

        self.conv3x3 = nn.Conv2d(2*hidden_dim, hidden_dim, 3, padding=1)
        self.In = nn.InstanceNorm2d(hidden_dim, affine=True)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.1)
        self.seg_head =  nn.Conv2d(hidden_dim, 1, kernel_size=1)
    
    def forward(self, feats, word):

        sf0 = F.interpolate(input=feats[0], size=(self.img_size//4, self.img_size//4), mode='bilinear', align_corners=True)

        sf1 = F.interpolate(input=feats[1], size=(self.img_size//4, self.img_size//4), mode='bilinear', align_corners=True)
        sf2 = F.interpolate(input=feats[2], size=(self.img_size//4, self.img_size//4), mode='bilinear', align_corners=True)
        sf3 = F.interpolate(input=feats[3], size=(self.img_size//4, self.img_size//4), mode='bilinear', align_corners=True)
        seg_f = self.conv1x1(torch.cat([sf0, sf1, sf2, sf3], dim=1))
        if self.fus:
            seg_f = self.fgop(seg_f, word)
        fs = self.conv3x3(seg_f)
        fs = self.In(fs)
        fs = self.LeakyReLU(fs)
        fs = self.dropout1(fs)
        mask_pre = self.seg_head(fs)
        seg_feats = torch.concat([fs, mask_pre], dim=1)
        mask_pre = F.interpolate(input=mask_pre, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)

        return mask_pre, seg_feats
        
class grasp(nn.Module):
    def __init__(self, in_dim, hidden_dim, seg_dim, img_size):
        super().__init__()
        self.img_size = img_size

        self.conv1x1 = nn.Conv2d(4*in_dim, 2*hidden_dim, kernel_size=1)

        self.conv3x3 = nn.Conv2d(2*hidden_dim, hidden_dim, 3, padding=1)
        self.bn = nn.InstanceNorm2d(hidden_dim, affine=True)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv1x1_2 = nn.Conv2d(hidden_dim + seg_dim + 1, hidden_dim, 1)
        self.conv3x3_2 = nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1)
        self.bn_2 = nn.InstanceNorm2d(hidden_dim//2, affine=True)
        self.LeakyReLU_2 = nn.LeakyReLU(inplace=True)

        self.dropout2 = nn.Dropout(p=0.1)


        self.pos_output =  nn.Conv2d(hidden_dim//2, 1, kernel_size=1)
        self.cos_output =  nn.Conv2d(hidden_dim//2, 1, kernel_size=1)
        self.sin_output = nn.Conv2d(hidden_dim//2, 1, kernel_size=1)
        self.width_output = nn.Conv2d(hidden_dim//2, 1, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feats, seg_f):

        sf0 = F.interpolate(input=feats[0], size=(self.img_size//4, self.img_size//4), mode='bilinear', align_corners=True)
        sf1 = F.interpolate(input=feats[1], size=(self.img_size//4, self.img_size//4), mode='bilinear', align_corners=True)
        sf2 = F.interpolate(input=feats[2], size=(self.img_size//4, self.img_size//4), mode='bilinear', align_corners=True)
        sf3 = F.interpolate(input=feats[3], size=(self.img_size//4, self.img_size//4), mode='bilinear', align_corners=True)
        grasp_f = self.conv1x1(torch.cat([sf0, sf1, sf2, sf3], dim=1))

        grasp_f = self.conv3x3(grasp_f)
        grasp_f = self.bn(grasp_f)
        grasp_f = self.LeakyReLU(grasp_f)
        grasp_f = self.dropout1(grasp_f)

        grasp_f = self.conv1x1_2(torch.concat([grasp_f, seg_f], dim=1))

        grasp_f = F.interpolate(input=grasp_f, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)

        grasp_f = self.conv3x3_2(grasp_f)
        grasp_f = self.bn_2(grasp_f)
        grasp_f = self.LeakyReLU_2(grasp_f)
        grasp_f = self.dropout2(grasp_f)


        pos = self.pos_output(grasp_f)
        cos = self.cos_output(grasp_f)
        sin = self.sin_output(grasp_f)
        width = self.width_output(grasp_f)
        
        pos = torch.sigmoid(pos)
        width = torch.sigmoid(width)

        return pos, cos, sin, width


class dualDecoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.neck = neck(args.feats_dims, args.word_dim, hidden_dim=2*args.seg_dim)
        self.seg = segmentation(2*args.seg_dim, hidden_dim=args.seg_dim, img_size=args.img_size, word_dim=args.word_dim)
        self.grasp_head = grasp(2*args.seg_dim, args.grasp_dim, seg_dim=args.seg_dim, img_size=args.img_size)
    
    def forward(self, features, word, mask=None):
        feats = self.neck(features, word, mask)
        seg_mask, seg_feats= self.seg(feats, word)
        pos, cos, sin, width = self.grasp_head(feats, seg_feats)

        return seg_mask, pos, cos, sin, width



import torch
import torch.nn as nn
# from .attention_cross import crossAttn
from .attention import CBAM, crossAttn

class ResCBAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=16):
        
        super(ResCBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.In1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.In2 = nn.BatchNorm2d(out_channels)
        self.cbam = CBAM(out_channels, ratio=ratio)
        self.use_1x1conv = False
        if in_channels != out_channels:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.use_1x1conv = True

        self.LeakyReLU2 = nn.LeakyReLU(inplace=True)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x_in):
        x = self.In1(self.conv1(x_in))
        x = self.In2(self.conv2(x))
        x = self.cbam(x)
        if self.use_1x1conv:
            x_in = self.conv1x1(x_in)
        return self.LeakyReLU2(x + x_in)
    
class feed_forward(nn.Module):
    def __init__(self, in_dim, expand_ratio=4) -> None:
        super().__init__()
        hidden_dim = in_dim * expand_ratio

        self.linear = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
                                    nn.BatchNorm2d(hidden_dim),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
                                    nn.BatchNorm2d(hidden_dim),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(hidden_dim, in_dim, kernel_size=1),
                                    nn.BatchNorm2d(in_dim),
                                    )
        self.LR = nn.LeakyReLU(inplace=True)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
                
    def forward(self, img):
        residual_img = img
        img = self.linear(img)
        img = self.LR(img + residual_img)
        return img

class FGOP(nn.Module):
    def __init__(self, in_dim, out_dim, word_dim, ratio=16) -> None:
        super().__init__()
        self.rescbam = ResCBAM(in_dim, out_dim, ratio=ratio)
        self.CGOP = crossAttn(out_dim, word_dim)
        
        self.ff = feed_forward(out_dim)

    def forward(self, img, word, word_mask=None):
        out = self.rescbam(img)
        out = self.CGOP(out, word, word_mask)
        out = self.ff(out)
        
        
        return out

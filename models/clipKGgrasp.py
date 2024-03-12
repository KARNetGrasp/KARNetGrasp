import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip import build_model

from .decoder import dualDecoder
from .KAmodule import KAug

class KGgrasp(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_kg = args.use_kg
        clip_model = torch.jit.load(args.clip_pretrain, map_location="cpu").eval()
        
        self.encoder = build_model(clip_model.state_dict(), txt_length=args.txt_length, fus=False, frozen=False).float()
        self.decoder = dualDecoder(args)
        if self.use_kg:
            self.ka = KAug(word_dim=args.word_dim, num_layer=1)


    def forward(self, images, words, kg_word_embedding, mask=None):

        B, _, H, W = images.shape
        if mask is None:
            mask = torch.zeros_like(words).bool()

            if self.use_kg:
                kg_word = kg_word_embedding.mean(dim=-1)
                kg_mask = torch.zeros_like(kg_word).masked_fill_(kg_word == 0, 1).bool()
                kg_mask = torch.concat([mask, kg_mask], dim=-1).bool()
                mask = kg_mask

        image_features, word_features, logits_per_image, sent = self.encoder(images, words)
        if self.use_kg:
            word_features = self.ka(word_features, kg_word_embedding, mask)

        seg_mask, pos, cos, sin, width = self.decoder(image_features[::-1], word_features, mask)
        
        return seg_mask, pos, cos, sin, width
    

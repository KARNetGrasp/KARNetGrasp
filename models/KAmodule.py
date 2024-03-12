import torch
import torch.nn as nn

class KAug(nn.Module):
    def __init__(self, word_dim, num_layer=1) -> None:
        super().__init__()

        self.num_layer = num_layer

        self.kg_txt_embeding = nn.Linear(word_dim, word_dim)
        self.kg_norm = nn.LayerNorm(word_dim)
        self.selfAttn_list = nn.ModuleList([nn.MultiheadAttention(word_dim, num_heads=8)] * self.num_layer)
        self.feed_forward_list = nn.ModuleList([nn.Sequential(
            nn.Linear(word_dim, word_dim*4),
            nn.GELU(),
            nn.Linear(4*word_dim, word_dim)
        )] * self.num_layer)

        self.LN1 = nn.ModuleList([nn.LayerNorm(word_dim)] * self.num_layer)
        self.LN2 = nn.ModuleList([nn.LayerNorm(word_dim)] * self.num_layer)

    
    def forward(self, word, kg_word_embedding, mask=None):
        kg_word_embedding = self.kg_txt_embeding(kg_word_embedding)
        kg_word_embedding = self.kg_norm(kg_word_embedding)
        word_aug = torch.concat([word, kg_word_embedding], dim=1) # [B, 21, C]
        word_aug = word_aug.permute(1, 0, 2) # [21, B, C]
        # print("word aug:", word_aug.shape)
        for i in range(self.num_layer):
            residual_word1 = word_aug
            word_aug = self.LN1[i](word_aug)
            word_aug = self.selfAttn_list[i](word_aug, word_aug, word_aug, key_padding_mask=mask)[0]
            word_aug = word_aug + residual_word1

            residual_word2 = word_aug
            word_aug = self.LN2[i](word_aug)
            word_aug = self.feed_forward_list[i](word_aug)
            word_aug = word_aug + residual_word2

        word_aug = word_aug.permute(1, 0, 2) # [B, 21, C]
        return word_aug




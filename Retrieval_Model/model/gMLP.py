import torch.nn as nn
import einops
import torch


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()

        self.norm = nn.LayerNorm(d_ffn // 2)
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = v.permute(0, 2, 1)
        v = self.proj(v)
        v = v.permute(0, 2, 1)
        return u * v


class GatingMlpBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len, survival_prob):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.proj_1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.GELU()
        self.spatial_gating_unit = SpatialGatingUnit(d_ffn, seq_len)
        self.proj_2 = nn.Linear(d_ffn // 2, d_model)
        self.prob = survival_prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))

    def forward(self, x):
        if self.training and torch.equal(self.m.sample(), torch.zeros(1)):
            return x
        shorcut = x.clone()
        x = self.norm(x)
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shorcut


class gMLP(nn.Module):
    def __init__(
        self,
        d_model,
        d_ffn,
        seq_len,
        n_blocks,
        prob_0_L=[1, 0.5],
    ):
        super().__init__()

        self.survival_probs = torch.linspace(prob_0_L[0], prob_0_L[1], n_blocks)
        self.blocks = nn.ModuleList(
            [GatingMlpBlock(d_model, d_ffn, seq_len, prob) for prob in self.survival_probs]
        )

    def forward(self, x):
        for gmlp_block in self.blocks:
            x = gmlp_block(x)
        return x


class VisiongMLP(nn.Module):
    def __init__(
        self,
        image_size,
        n_channels,
        patch_size,
        d_model,
        d_ffn,
        n_blocks,
        n_classes,
        prob_0_L=[1, 0],
    ):
        super().__init__()

        image_height, image_width = image_size[0], image_size[1]
        patch_height, patch_width = patch_size, patch_size
        self.patch_size = patch_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0, 'image height and width must be divisible by patch size'
        self.n_patches = (image_height // patch_height) * (image_width // patch_width)
        self.seq_len = self.n_patches + 1

        self.patch_embedding = nn.Linear(n_channels * patch_size ** 2, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.gmlp = gMLP(d_model, d_ffn, self.seq_len, n_blocks, prob_0_L)
        self.representation = nn.Sequential(nn.Linear(d_model, 128),
                                             nn.Tanh())
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        n_samples = x.shape[0]

        x = einops.rearrange(
            x, "n c (h p1) (w p2) -> n (h w) (c p1 p2)", p1=self.patch_size, p2=self.patch_size
        )
        x = self.patch_embedding(x)

        cls_token = self.cls_token.expand(n_samples, 1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = self.gmlp(x)
        cls_token_final = x[:, 0]
        x = self.representation(cls_token_final)
        out = self.head(x)
        return x, out
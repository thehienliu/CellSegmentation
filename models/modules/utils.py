import torch
from torch import nn
from einops import rearrange

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 256.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # B H*W C
        return self.norm(x)

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
      """ Patch Merging Layer.

      Args:
          input_resolution (tuple[int]): Resolution of input feature.
          dim (int): Number of input channels.
          norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
      """

      super().__init__()
      self.dim = dim
      self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
      self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, expand_dim=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, expand_dim*dim, bias=False)
        self.norm = norm_layer(expand_dim*dim // (self.dim_scale ** 2))

    def forward(self, x):
        """
        x: B, C, H, W
        """
        x = x.permute(0, 2, 3, 1)
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale ** 2))
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

        return x
from torch import nn
from typing import List
from mamba_ssm import Mamba
from .utils import Permute, PatchEmbed, PatchMerging, PatchExpand

class Mamba2dBlock(nn.Module):
  def __init__(self, d_model, d_state, d_conv, expand, dropout=0.0, scan="bscan"):
    super().__init__()

    self.mamba = Mamba(d_model=d_model,
                       d_state=d_state,
                       d_conv=d_conv,
                       expand=expand)

    self.out_norm = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
    self.out_proj = nn.Linear(d_model, d_model)
    self.scan_method = scan

  def bscan_forward(self, x):
    # x shape b l d
    (b, l, d) = x.shape
    hw = int(l**0.5)

    x1 = x.view(b, hw, hw, d) # b h w d
    x2 = x1.flip(dims=[1]).transpose(1, 2) # b w h d

    # Flatten
    x1 = x1.view(b, -1, d) # b hw d
    x2 = x2.contiguous().view(b, -1, d) # b wh d

    y1 = self.mamba(x1)
    y2 = self.mamba(x2)
    y3 = self.mamba(x1.flip(dims=[1]))
    y4 = self.mamba(x2.flip(dims=[1]))

    # Transpose back
    y1 = y1.view(b, hw, hw, d) # b h w d
    y2 = y2.view(b, hw, hw, d).transpose(1, 2).flip(dims=[1]) # b w h d -> b h w d
    y3 = y3.flip(dims=[1]).view(b, hw, hw, d) # b h w d
    y4 = y4.flip(dims=[1]).view(b, hw, hw, d).transpose(1, 2).flip(dims=[1]) # b h w d

    y = y1 + y2 + y3 + y4 # b h w d -> Consider using a Linear to aggregate the the Mamba output
    out = self.out_norm(y).view(b, -1, d)

    return x + self.dropout(self.out_proj(out)) # Skip connection i test below is here

  def crossscan_forward(self, x):
    # x shape b l d
    (b, l, d) = x.shape
    hw = int(l**0.5)

    x1 = x.view(b, hw, hw, d) # b h w d
    x2 = x1.clone().transpose(1, 2) # b w h d

    # Flatten
    x1 = x1.view(b, -1, d) # b hw d
    x2 = x2.contiguous().view(b, -1, d) # b wh d

    y1 = self.mamba(x1)
    y2 = self.mamba(x2)
    y3 = self.mamba(x1.flip(dims=[1]))
    y4 = self.mamba(x2.flip(dims=[1]))

    # Transpose back
    y1 = y1.view(b, hw, hw, d) # b h w d
    y2 = y2.view(b, hw, hw, d).transpose(1, 2).flip(dims=[1]) # b w h d -> b h w d
    y3 = y3.flip(dims=[1]).view(b, hw, hw, d) # b h w d
    y4 = y4.flip(dims=[1]).view(b, hw, hw, d).transpose(1, 2).flip(dims=[1]) # b h w d

    y = y1 + y2 + y3 + y4 # b h w d -> Consider using a Linear to aggregate the the Mamba output
    out = self.out_norm(y).view(b, -1, d)

    return x + self.dropout(self.out_proj(out)) # Skip connection i test below is here

  def forward(self, x):

    if self.scan_method == "cross-scan":
      return self.crossscan_forward(x)

    return self.bscan_forward(x)
    

class Mamba2dLayer(nn.Module):
  def __init__(self, d_model, d_state, d_conv, expand, depth, dropout=0.0):
    super().__init__()

    self.blocks = nn.ModuleList()
    self.norms = nn.ModuleList()

    for i in range(depth):
      self.blocks.append(Mamba2dBlock(d_model, d_state, d_conv, expand))
      self.norms.append(nn.LayerNorm(d_model) if i < depth - 1 else nn.Identity())

  def forward(self, x):

    for blk, norm in zip(self.blocks, self.norms):
      x = norm(blk(x))

    (b, l, d) = x.shape
    hw = int(l ** 0.5)

    return x.view(b, hw, hw, d)


class BasicLayer_up(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, depth, dropout=0.1, dim_scale=2):

        super().__init__()

        # build blocks
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(depth):
          self.blocks.append(Mamba2dBlock(d_model, d_state, d_conv, expand))
          self.norms.append(nn.LayerNorm(d_model))

        # patch merging layer
        self.upsample = PatchExpand(dim=d_model, dim_scale=dim_scale)

    def forward(self, x):

        for blk, norm in zip(self.blocks, self.norms):
            x = self.dropout(norm(blk(x)))

        (b, l, d) = x.shape
        hw = int(l ** 0.5)

        x = self.upsample(x.view(b, hw, hw, d))
        return x


class ImageEncoder(nn.Module):
  def __init__(self,
               num_classes: int,
               extract_layers: List[int] = [1, 2, 3],
               in_chans: int = 3,
               dims: List[int] = [96, 192, 384, 768], # dims must be double every step
               patch_size: int = 4,
               depth: List[int] = [2, 2, 5, 2],
               d_state: int = 16,
               d_conv: int = 3,
               expand: int = 2):
    """
    Args:
        in_chans (int): Input channel of the image
        patch_size (int): Patch size, decide the image size after patch embed B C H W -> B C H/patch_size W/patch_size
        depth (int): Number of the Mamba block
        d_state (int): The dimension size of B and C in Mamba
        d_conv (int): The kernel size of Conv1d in Mamba

    """
    super().__init__()
    self.extract_layers = extract_layers
    self.dims = dims

    self.patch_embed = PatchEmbed(in_chans=in_chans,
                                  embed_dim=dims[0],
                                  patch_size=patch_size)

    self.blocks = nn.ModuleList()
    self.downsamples = nn.ModuleList()

    for i in range(len(depth)):
      block = Mamba2dLayer(d_model=dims[i],
                           d_state=d_state,
                           d_conv=d_conv,
                           expand=expand,
                           depth=depth[i])

      down = PatchMerging(dims[i]) if i < len(depth) - 1 else nn.Identity()

      self.blocks.append(block)
      self.downsamples.append(down)

    self.classifier = nn.Sequential(
            nn.LayerNorm(dims[len(depth) - 1]), # B,H,W,C
            Permute(0, 3, 1, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(dims[len(depth) - 1], num_classes),
        )

  def forward(self, x):

    b = x.shape[0]

    extracted_layers = []
    x = self.patch_embed(x)

    for depth, (blk, down) in enumerate(zip(self.blocks, self.downsamples)):
      x = blk(x) # x: B H W C

      if depth + 1 in self.extract_layers:
        extracted_layers.append(x)

      x = down(x)

    return self.classifier(x), x, extracted_layers
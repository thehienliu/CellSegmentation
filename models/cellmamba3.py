import torch
import numpy as np
from torch import nn
from typing import List
from mamba_ssm import Mamba
from einops import rearrange
from functools import partial
from collections import OrderedDict
from timm.models.layers import DropPath
from models.modules.mamba import BasicLayer_up
from typing import Optional, Tuple, Type, List, Dict
from models.modules.cnn import Conv2DBlock, Deconv2DBlock
from utils.post_proc_cellmamba import DetectionCellPostProcessor
from CellViT.models.encoders.VIT.SAM.image_encoder import ImageEncoderViT


# model path https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
class ViTCellViTDeit(ImageEncoderViT):
    def __init__(
        self,
        extract_layers: List[int],
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            out_chans,
            qkv_bias,
            norm_layer,
            act_layer,
            use_abs_pos,
            use_rel_pos,
            rel_pos_zero_init,
            window_size,
            global_attn_indexes,
        )
        self.extract_layers = extract_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extracted_layers = []
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            token_size = x.shape[1]
            x = x + self.pos_embed[:, :token_size, :token_size, :]

        for depth, blk in enumerate(self.blocks):
            x = blk(x)
            if depth + 1 in self.extract_layers:
                extracted_layers.append(x)
        output = self.neck(x.permute(0, 3, 1, 2))
        _output = rearrange(output, "b c h w -> b c (h w)")

        return torch.mean(_output, axis=-1), output, extracted_layers


class CellMamba(nn.Module):
    def __init__(self,
                 num_classes: int,
                 drop_rate: float = 0.2,
                 num_nuclei_classes: int = 6,
                 extract_layers: List[int] = [1, 2, 3],
                 in_chans: int = 3,
                 dims: List[int] = [96, 192, 384, 768], # dims must be double every step
                 patch_size: int = 4,
                 depth: List[int] = [2, 2, 2, 2],
                 d_state: int = 16,
                 d_conv: int = 3,
                 expand: int = 2,
                 classifier_dim: int = 256):

        '''
        Args:
            num_classes (int): Number of tissue class
            drop_rate (float): Dropout probability
            num_nuclei_classes (int): number of nuclei class
            extract_layers (List[int]): extract at which layers
            in_chans (int): Input channel of the image
            dims (int): num features at each layer
            patch_size (int): Patch size, decide the image size after patch embed B C H W -> B C H/patch_size W/patch_size
            depth (int): Number of the Mamba block
            d_state (int): The dimension size of B and C in Mamba
            d_conv (int): The kernel size of Conv1d in Mamba
            drop_rate (float): drop rate
        '''

        super().__init__()


        self.dims = list(reversed(dims)) # Reverse the dim
        self.drop_rate = drop_rate
        self.num_nuclei_classes = num_nuclei_classes

        # Skip connection
        self.concat_linear1 = nn.Sequential(
            nn.Linear(self.dims[2], self.dims[3]),
            nn.LayerNorm(self.dims[3])
            )
        
        self.concat_linear2 = nn.Sequential(
            nn.Linear(self.dims[1], self.dims[2]),
            nn.LayerNorm(self.dims[2])
            )
        self.concat_linear3 = nn.Sequential(
            nn.Linear(self.dims[0], self.dims[1]),
            nn.LayerNorm(self.dims[1])
            )


        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 48, 3, dropout=self.drop_rate),
            Conv2DBlock(48, self.dims[3], 3, dropout=self.drop_rate),
        )  # skip connection after positional encoding, shape should be H, W, 64
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(768, self.dims[2], dropout=self.drop_rate),
            Deconv2DBlock(self.dims[2], self.dims[3], dropout=self.drop_rate),
            Deconv2DBlock(self.dims[3], self.dims[3], dropout=self.drop_rate),
        )  # skip connection 1
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(768, self.dims[1], dropout=self.drop_rate),
            Deconv2DBlock(self.dims[1], self.dims[2], dropout=self.drop_rate),
        )  # skip connection 2
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(768, self.dims[1], dropout=self.drop_rate)
        )  # skip connection 3


        # Build downsampler
        self.encoder = ViTCellViTDeit(extract_layers=[3, 6, 9, 12],
                                      depth=12,
                                      embed_dim=768,
                                      mlp_ratio=4,
                                      norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                                      num_heads=12,
                                      qkv_bias=True,
                                      use_rel_pos=True,
                                      global_attn_indexes=[2, 5, 8, 11],
                                      window_size=14,
                                      out_chans=256,)
        
        self.classifier = nn.Linear(classifier_dim, num_classes)

        # Build upsampler
        self.nuclei_binary_map_decoder = self.create_upsampling_branch(num_classes=2, depth=depth)
        self.hv_map_decoder = self.create_upsampling_branch(num_classes=2, depth=depth)
        self.nuclei_type_maps_decoder = self.create_upsampling_branch(num_classes=self.num_nuclei_classes, depth=depth)

    def _strip_state_dict(self, state_dict: Dict) -> OrderedDict:
        """Strip the 'image_encoder' from the SAM state dict keys."""
        new_dict = {}
        for k, w in state_dict.items():
            if "image_encoder" in k:
                spl = ["".join(kk) for kk in k.split(".")]
                new_key = ".".join(spl[1:])
                new_dict[new_key] = w

        return new_dict

    def load_pretrained_encoder(self, ckpt_path) -> None:
        """Load the weights from the checkpoint."""
        state_dict = torch.load(
            ckpt_path, map_location=lambda storage, loc: storage
        )
        try:
            msg = self.encoder.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            new_ckpt = self._strip_state_dict(state_dict)
            msg = self.encoder.load_state_dict(new_ckpt, strict=True)
        except BaseException as e:
            raise RuntimeError(f"Error loading checkpoint: {e}")

        print(f"Loading pre-trained {type(self.encoder).__name__} checkpoint: {msg}")

    def forward(self, x):
        ''' Forward pass
        Args:
            x (torch.Tensor): B C H W

        Returns:
            dict: Output for all branches
                * tissue_types: Raw tissue type prediction. Shape: (B, num_tissue_classes)
                * nuclei_binary_map: Raw binary cell segmentation predictions. Shape: (B, 2, H, W)
                * hv_map: Binary HV Map predictions. Shape: (B, 2, H, W)
                * nuclei_type_map: Raw binary nuclei type preditcions. Shape: (B, num_nuclei_classes, H, W)
                * retrieve tokens: tokens
        '''

        out_dict = {}

        classifier_logits, z_, z = self.encoder(x)
        out_dict['tissue_types'] = self.classifier(classifier_logits)

        z0, z1, z2, z3, z4 = x, *z

        z4 = z4.permute(0, 3, 1, 2)
        z3 = z3.permute(0, 3, 1, 2)
        z2 = z2.permute(0, 3, 1, 2)
        z1 = z1.permute(0, 3, 1, 2)

        # Cell segmentation branch
        out_dict["nuclei_binary_map"] = self._forward_upsample(z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder)
        out_dict["hv_map"] = self._forward_upsample(z0, z1, z2, z3, z4, self.hv_map_decoder)
        out_dict["nuclei_type_map"] = self._forward_upsample(z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder)

        return out_dict


    def _forward_upsample(self,
                          z0: torch.Tensor,
                          z1: torch.Tensor,
                          z2: torch.Tensor, # z2 shape b 32 32 192
                          z3: torch.Tensor, # z3 shape b 16 16 384
                          z4: torch.Tensor,
                          branch_decoder: nn.Sequential,
                      ) -> torch.Tensor:

        b = z4.shape[0]
        b4 = branch_decoder.bottleneck_upsampler(z4.view(b, -1, self.dims[0])) # z4 shape b 8 8 768 -> b4 shape b 16 16 384
        
        z3 = self.decoder3(z3).permute(0, 2, 3, 1)
        b3 = self.concat_linear3(torch.cat([z3, b4], dim=-1)) # b3 shape b 16 16 384
        b3 = branch_decoder.decoder3_upsampler(b3.view(b, -1, self.dims[1])) # b3 shape b 32 32 192
        
        z2 = self.decoder2(z2).permute(0, 2, 3, 1)
        b2 = self.concat_linear2(torch.cat([z2, b3], dim=-1)) # b2 shape b 32 32 192
        b2 = branch_decoder.decoder2_upsampler(b2.view(b, -1, self.dims[2])) # b2 shape b 64 64 96
        
        z1 = self.decoder1(z1).permute(0, 2, 3, 1)
        b1 = self.concat_linear1(torch.cat([z1, b2], dim=-1)) # b1 shape b 64 64 96
        b1 = branch_decoder.decoder1_upsampler(b1.view(b, -1, self.dims[3])) # b1 shape b 256 256 24

        branch_output = branch_decoder.decoder0_header(b1.permute(0, 3, 1, 2))
        return branch_output


    def create_upsampling_branch(self, num_classes, depth):

        bottleneck_upsampler = BasicLayer_up(d_model=self.dims[0],
                                             d_state=16,
                                             d_conv=3,
                                             expand=2,
                                             depth=depth[3])
        
        upsampler_3 = BasicLayer_up(d_model=self.dims[1],
                                    d_state=16,
                                    d_conv=3,
                                    expand=2,
                                    depth=depth[2])
        
        upsampler_2 = BasicLayer_up(d_model=self.dims[2],
                                    d_state=16,
                                    d_conv=3,
                                    expand=2,
                                    depth=depth[1])
        
        upsampler_1 = BasicLayer_up(d_model=self.dims[3],
                                    d_state=16,
                                    d_conv=3,
                                    expand=2,
                                    depth=depth[0],
                                    dim_scale=2)
        
        upsampler_0 = nn.Conv2d(in_channels=self.dims[3]//2, out_channels=num_classes, kernel_size=1, bias=False)

        decoder = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", upsampler_3),
                    ("decoder2_upsampler", upsampler_2),
                    ("decoder1_upsampler", upsampler_1),
                    ("decoder0_header", upsampler_0),
                ]
            )
        )

        return decoder


    def calculate_instance_map(self, predictions, magnification = 40):
        """Calculate Instance Map from network predictions (after Softmax output)

        Args:
            predictions (dict): Dictionary with the following required keys:
                * nuclei_binary_map: Binary Nucleus Predictions. Shape: (B, 2, H, W)
                * nuclei_type_map: Type prediction of nuclei. Shape: (B, self.num_nuclei_classes, H, W)
                * hv_map: Horizontal-Vertical nuclei mapping. Shape: (B, 2, H, W)
            magnification (Literal[20, 40], optional): Which magnification the data has. Defaults to 40.

        Returns:
            Tuple[torch.Tensor, List[dict]]:
                * torch.Tensor: Instance map. Each Instance has own integer. Shape: (B, H, W)
                * List of dictionaries. Each List entry is one image. Each dict contains another dict for each detected nucleus.
                    For each nucleus, the following information are returned: "bbox", "centroid", "contour", "type_prob", "type"
        """
        # reshape to B, H, W, C
        predictions_ = predictions.copy()
        predictions_["nuclei_type_map"] = predictions_["nuclei_type_map"].permute(0, 2, 3, 1)
        predictions_["nuclei_binary_map"] = predictions_["nuclei_binary_map"].permute(0, 2, 3, 1)
        predictions_["hv_map"] = predictions_["hv_map"].permute(0, 2, 3, 1)

        cell_post_processor = DetectionCellPostProcessor(
            nr_types=self.num_nuclei_classes, magnification=magnification, gt=False
        )
        instance_preds = []
        type_preds = []

        for i in range(predictions_["nuclei_binary_map"].shape[0]):
            pred_map = np.concatenate(
                [
                    torch.argmax(predictions_["nuclei_type_map"], dim=-1)[i]
                    .detach()
                    .cpu()[..., None],
                    torch.argmax(predictions_["nuclei_binary_map"], dim=-1)[i]
                    .detach()
                    .cpu()[..., None],
                    predictions_["hv_map"][i].detach().cpu(),
                ],
                axis=-1,
            )
            instance_pred = cell_post_processor.post_process_cell_segmentation(pred_map)
            instance_preds.append(instance_pred[0])
            type_preds.append(instance_pred[1])

        return torch.Tensor(np.stack(instance_preds)), type_preds

    def generate_instance_nuclei_map(self, instance_maps: torch.Tensor, type_preds: List[dict]):
        """Convert instance map (binary) to nuclei type instance map

        Args:
            instance_maps (torch.Tensor): Binary instance map, each instance has own integer. Shape: (B, H, W)
            type_preds (List[dict]): List (len=B) of dictionary with instance type information (compare post_process_hovernet function for more details)

        Returns:
            torch.Tensor: Nuclei type instance map. Shape: (B, self.num_nuclei_classes, H, W)
        """
        batch_size, h, w = instance_maps.shape
        instance_type_nuclei_maps = torch.zeros(
            (batch_size, h, w, self.num_nuclei_classes)
        )
        for i in range(batch_size):
            instance_type_nuclei_map = torch.zeros((h, w, self.num_nuclei_classes))
            instance_map = instance_maps[i]
            type_pred = type_preds[i]
            for nuclei, spec in type_pred.items():
                nuclei_type = spec["type"]
                instance_type_nuclei_map[:, :, nuclei_type][
                    instance_map == nuclei
                ] = nuclei

            instance_type_nuclei_maps[i, :, :, :] = instance_type_nuclei_map

        instance_type_nuclei_maps = instance_type_nuclei_maps.permute(0, 3, 1, 2)
        return torch.Tensor(instance_type_nuclei_maps)
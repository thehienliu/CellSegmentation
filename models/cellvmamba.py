import torch
import numpy as np
from torch import nn
from typing import List
from collections import OrderedDict
from models.vmamba.vmamba import VSSM
from models.modules.cnn import Conv2DBlock
from utils.post_proc_cellmamba import DetectionCellPostProcessor

class CellVMamba(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_nuclei_classes: int,
                 pretrained_weight: str = None,

                 in_chans: int = 3,
                 imgsize: int = 256,
                 drop_rate: float = 0.1,

                 dims: List[int] = [96, 192, 384, 768],
                 depths: List[int] = [2, 2, 8, 2],
                 ssm_d_state: int = 1,
                 ssm_dt_rank: str = 'auto',
                 ssm_ratio: float = 1.0,
                 ssm_conv: int = 3,
                 ssm_conv_bias: bool = False,
                 forward_type: str = "v05_noz",
                 mlp_ratio: float = 4.0,

                 patch_size: int = 4,
                 drop_path_rate: float = 0.2,
                 ) -> None:
      super().__init__()

      self.dims = dims
      self.drop_rate = drop_rate
      self.num_nuclei_classes = num_nuclei_classes
      self.encoder = VSSM(patch_size=patch_size,
                          in_chans=in_chans,
                          num_classes=num_classes,
                          depths=depths,
                          dims=dims,
                          # ====================
                          ssm_init="v0",
                          ssm_drop_rate=0.0,
                          ssm_rank_ratio=2.0,
                          ssm_conv=ssm_conv,
                          ssm_ratio=ssm_ratio,
                          ssm_dt_rank=ssm_dt_rank,
                          ssm_d_state=ssm_d_state,
                          forward_type=forward_type,
                          ssm_conv_bias=ssm_conv_bias,
                          # ====================
                          mlp_ratio=mlp_ratio,
                          mlp_drop_rate=0.0,
                          # ====================
                          drop_path_rate=drop_path_rate,
                          patch_norm=True,
                          gmlp=False,
                          use_checkpoint=False,
                          # ===================
                          posembed=False,
                          imgsize=imgsize)
      
      if pretrained_weight:
        self.encoder.load_pretrained(pretrained_weight)

      self.decoder0 = nn.Sequential(
            Conv2DBlock(in_chans, dims[0] // 2, dropout=self.drop_rate)
      )

      self.decoder1 = nn.Sequential(
            Conv2DBlock(dims[0] // 2, dims[0] // 2, dropout=self.drop_rate)
      )

      self.decoder2 = nn.Sequential(
            Conv2DBlock(dims[0], dims[0], dropout=self.drop_rate)
      )

      self.decoder3 = nn.Sequential(
            Conv2DBlock(dims[1], dims[1], dropout=self.drop_rate)
      )

      self.decoder4 = nn.Sequential(
            Conv2DBlock(dims[2], dims[2], dropout=self.drop_rate)
      )

      # Build upsampler
      self.nuclei_binary_map_decoder = self.create_upsampling_branch(num_classes=2)
      self.hv_map_decoder = self.create_upsampling_branch(num_classes=2)
      self.nuclei_type_maps_decoder = self.create_upsampling_branch(num_classes=num_nuclei_classes)


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

        classifier_logits, z = self.encoder(x)
        out_dict['tissue_types'] = classifier_logits

        z0, z1, z2, z3, z4, z5 = x, *z # 256 128 64 32 16 8

        # Cell segmentation branch
        out_dict["nuclei_binary_map"] = self._forward_upsample(z0, z1, z2, z3, z4, z5, self.nuclei_binary_map_decoder)
        out_dict["hv_map"] = self._forward_upsample(z0, z1, z2, z3, z4, z5, self.hv_map_decoder)
        out_dict["nuclei_type_map"] = self._forward_upsample(z0, z1, z2, z3, z4, z5, self.nuclei_type_maps_decoder)

        return out_dict

    def _forward_upsample(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
        z5: torch.Tensor,
        branch_decoder: nn.Sequential,
    ) -> torch.Tensor:
        """Forward upsample branch

        Args:
            z0 (torch.Tensor): Highest skip
            z1 (torch.Tensor): 1. Skip
            z2 (torch.Tensor): 2. Skip
            z3 (torch.Tensor): 3. Skip
            z4 (torch.Tensor): 4. Skip
            z5 (torch.Tensor): Bottleneck
            branch_decoder (nn.Sequential): Branch decoder network

        Returns:
            torch.Tensor: Branch Output
        """
        # z5 - 32 768 8 8
        # z4 - 32 384 16 16
        # z3 - 32 192 32 32
        # z2 - 32 96 64 64
        # z1 - 32 48 128 128
        # z0 - 32 3 256 256


        b5 = branch_decoder.bottleneck_upsampler(z5) # b5 - 32 384 16 16

        b4 = self.decoder4(z4) # b4 - 32 384 16 16
        b4 = branch_decoder.decoder3_upsampler(torch.cat([b4, b5], dim=1)) # b4 - 32 192 32 32

        b3 = self.decoder3(z3) # b3 - 32 192 32 32
        b3 = branch_decoder.decoder2_upsampler(torch.cat([b3, b4], dim=1)) # b3 - 32 96 64 64

        b2 = self.decoder2(z2) # b2 - 32 96 64 64
        b2 = branch_decoder.decoder1_upsampler(torch.cat([b2, b3], dim=1)) # b2 - 32 48 128 128

        b1 = self.decoder1(z1) # b0 - 32 48 128 128
        b1 = branch_decoder.decoder0_upsampler(torch.cat([b1, b2], dim=1)) # b1 - 32 48 256 256

        b0 = self.decoder0(z0) # b0 - 32 48 256 256
        branch_output = branch_decoder.header(torch.cat([b0, b1], dim=1))

        return branch_output

    def create_upsampling_branch(self, num_classes: int) -> nn.Module:
        """Create Upsampling branch

        Args:
            num_classes (int): Number of output classes

        Returns:
            nn.Module: Upsampling path
        """

        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.dims[3],
            out_channels=self.dims[2],
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )

        # Upsampler 3
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(
                self.dims[2] * 2, self.dims[2], dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.dims[2], self.dims[2], dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.dims[2], self.dims[2], dropout=self.drop_rate
            ),
            nn.ConvTranspose2d(
                in_channels=self.dims[2],
                out_channels=self.dims[1],
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )

        # Upsampler 2
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(self.dims[1] * 2, self.dims[1], dropout=self.drop_rate),
            Conv2DBlock(self.dims[1], self.dims[1], dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=self.dims[1],
                out_channels=self.dims[0],
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )

        # Upsampler 1
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(self.dims[0] * 2, self.dims[0], dropout=self.drop_rate),
            Conv2DBlock(self.dims[0], self.dims[0], dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=self.dims[0],
                out_channels=self.dims[0] // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )

        # Upsampler 0
        decoder0_upsampler = nn.Sequential(
            Conv2DBlock(self.dims[0], self.dims[0], dropout=self.drop_rate),
            Conv2DBlock(self.dims[0], self.dims[0], dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=self.dims[0],
                out_channels=self.dims[0] // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )

        # Header
        header = nn.Sequential(
            Conv2DBlock(self.dims[0], self.dims[0], dropout=self.drop_rate),
            Conv2DBlock(self.dims[0], self.dims[0], dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=self.dims[0],
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        decoder = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                    ("decoder0_upsampler", decoder0_upsampler),
                    ("header", header),
                ]
            )
        )

        return decoder

    def freeze_encoder(self):

      for name, params in self.encoder.named_parameters():
        if "classifier.head" in name:
          continue
        params.requires_grad = False # params.requires_grad = "classifier.head" in name

    def unfreeze_encoder(self):

      for params in self.encoder.parameters():
        params.requires_grad = True

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
import torch
from torch import nn
import numpy as np
from mamba_ssm import Mamba
from typing import List
from collections import OrderedDict
from utils.post_proc_cellmamba import DetectionCellPostProcessor
from models.modules.cnn import Conv2DBlock, Deconv2DBlock
from models.modules.mamba import ImageEncoder

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
                 expand: int = 2):

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
        self.decoder0 = nn.Sequential(
            Conv2DBlock(in_chans, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.dims[3], self.dims[3], dropout=self.drop_rate),
        )
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.dims[2], self.dims[2], dropout=self.drop_rate),
        )
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.dims[1], self.dims[1], dropout=self.drop_rate)
        )

        # Build downsampler
        self.encoder = ImageEncoder(num_classes=num_classes,
                                    extract_layers=extract_layers,
                                    in_chans=in_chans,
                                    dims=dims,
                                    patch_size=patch_size,
                                    depth=depth,
                                    d_state=d_state,
                                    d_conv=d_conv,
                                    expand=expand,)

        # Build upsampler
        self.nuclei_binary_map_decoder = self.create_upsampling_branch(num_classes=2)
        self.hv_map_decoder = self.create_upsampling_branch(num_classes=2)
        self.nuclei_type_maps_decoder = self.create_upsampling_branch(num_classes=self.num_nuclei_classes)


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
        out_dict['tissue_types'] = classifier_logits

        z0, z1, z2, z3, z4 = x, *z, z_

        # Cell segmentation branch
        out_dict["nuclei_binary_map"] = self._forward_upsample(z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder)
        out_dict["hv_map"] = self._forward_upsample(z0, z1, z2, z3, z4, self.hv_map_decoder)
        out_dict["nuclei_type_map"] = self._forward_upsample(z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder)

        return out_dict


    def _forward_upsample(self,
                          z0: torch.Tensor,
                          z1: torch.Tensor,
                          z2: torch.Tensor,
                          z3: torch.Tensor,
                          z4: torch.Tensor,
                          branch_decoder: nn.Sequential,
                      ) -> torch.Tensor:

        b4 = branch_decoder.bottleneck_upsampler(z4)
        b3 = self.decoder3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b2 = self.decoder2(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        b1 = self.decoder1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        b0 = self.decoder0(z0)
        branch_output = branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))
        return branch_output


    def create_upsampling_branch(self, num_classes):

        # Bottleneck Upsampler
        bottleneck_upsampler = nn.Sequential(
            Deconv2DBlock(self.dims[0], self.dims[1], dropout=self.drop_rate),
            Deconv2DBlock(self.dims[1], self.dims[1], dropout=self.drop_rate),
        )

        # Upsampler 3
        upsampler_3 = nn.Sequential(
            Conv2DBlock(
                in_channels=self.dims[1]*2, out_channels=self.dims[1], dropout=self.drop_rate
            ),

            Conv2DBlock(
                in_channels=self.dims[1], out_channels=self.dims[1], dropout=self.drop_rate
            ),

            Conv2DBlock(
                in_channels=self.dims[1], out_channels=self.dims[1], dropout=self.drop_rate
            ),

            nn.ConvTranspose2d(
                in_channels=self.dims[1],
                out_channels=self.dims[2],
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )

        # Upsampler 2
        upsampler_2 = nn.Sequential(
            Conv2DBlock(
                in_channels=self.dims[2] * 2, out_channels=self.dims[2], dropout=self.drop_rate
            ),

            Conv2DBlock(
                in_channels=self.dims[2], out_channels=self.dims[2], dropout=self.drop_rate
            ),

            nn.ConvTranspose2d(
                in_channels=self.dims[2],
                out_channels=self.dims[3],
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )

        # Upsampler 1
        upsampler_1 = nn.Sequential(
            Conv2DBlock(
                in_channels=self.dims[3] * 2, out_channels=self.dims[3], dropout=self.drop_rate
            ),

            Conv2DBlock(
                in_channels=self.dims[3], out_channels=self.dims[3], dropout=self.drop_rate
            ),

            nn.ConvTranspose2d(
                in_channels=self.dims[3],
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )

        # Upsampler 0
        upsampler_0 = nn.Sequential(
            Conv2DBlock(
                in_channels=64 * 2, out_channels=64, dropout=self.drop_rate
            ),

            Conv2DBlock(
                in_channels=64, out_channels=64, dropout=self.drop_rate
            ),

            nn.Conv2d(
                in_channels=64,
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
    
from cell_segmentation.utils.base_loss import retrieve_loss_fn
from torch import nn
def get_loss_fn() -> dict:
        """Create a dictionary with loss functions for all branches

        Branches: "nuclei_binary_map", "hv_map", "nuclei_type_map", "tissue_types"

        Args:
            loss_fn_settings (dict): Dictionary with the loss function settings. Structure
            branch_name(str):
                loss_name(str):
                    loss_fn(str): String matching to the loss functions defined in the LOSS_DICT (base_ml.base_loss)
                    weight(float): Weighting factor as float value
                    (optional) args:  Optional parameters for initializing the loss function
                            arg_name: value

            If a branch is not provided, the defaults settings (described below) are used.

            For further information, please have a look at the file configs/examples/cell_segmentation/train_cellvit.yaml
            under the section "loss"

            Example:
                  nuclei_binary_map:
                    bce:
                        loss_fn: xentropy_loss
                        weight: 1
                    dice:
                        loss_fn: dice_loss
                        weight: 1

        Returns:
            dict: Dictionary with loss functions for each branch. Structure:
                branch_name(str):
                    loss_name(str):
                        "loss_fn": Callable loss function
                        "weight": weight of the loss since in the end all losses of all branches are added together for backward pass
                    loss_name(str):
                        "loss_fn": Callable loss function
                        "weight": weight of the loss since in the end all losses of all branches are added together for backward pass
                branch_name(str)
                ...

        Default loss dictionary:
            nuclei_binary_map:
                bce:
                    loss_fn: xentropy_loss
                    weight: 1
                dice:
                    loss_fn: dice_loss
                    weight: 1
            hv_map:
                mse:
                    loss_fn: mse_loss_maps
                    weight: 1
                msge:
                    loss_fn: msge_loss_maps
                    weight: 1
            nuclei_type_map
                bce:
                    loss_fn: xentropy_loss
                    weight: 1
                dice:
                    loss_fn: dice_loss
                    weight: 1
            tissue_types
                ce:
                    loss_fn: nn.CrossEntropyLoss()
                    weight: 1
        """
        loss_fn_dict = {}

        loss_fn_dict["nuclei_binary_map"] = {
                "focaltverskyloss": {"loss_fn": retrieve_loss_fn("FocalTverskyLoss"), "weight": 1},
                "dice": {"loss_fn": retrieve_loss_fn("dice_loss"), "weight": 1},}

        loss_fn_dict["hv_map"] = {
            "mse": {"loss_fn": retrieve_loss_fn("mse_loss_maps"), "weight": 2.5},
            "msge": {"loss_fn": retrieve_loss_fn("msge_loss_maps"), "weight": 8},
        }
        loss_fn_dict["nuclei_type_map"] = {
            "bce": {"loss_fn": retrieve_loss_fn("xentropy_loss"), "weight": 0.5},
            "dice": {"loss_fn": retrieve_loss_fn("dice_loss"), "weight": 0.2},
            "mcfocaltverskyloss": {"loss_fn": retrieve_loss_fn("MCFocalTverskyLoss", num_classes=6), "weight": 0.5},
        }
        loss_fn_dict["tissue_types"] = {
            "ce": {"loss_fn": nn.CrossEntropyLoss(), "weight": 0.1},
        }

        return loss_fn_dict
from models.cellmamba import CellMamba
from models.cellmamba2 import CellMambaV2
from models.cellmamba3 import CellMambaV3


def load_model(config, num_classes):
    if config.experiment == 'cellmamba':
        model = CellMamba(num_classes=num_classes)

    elif config.experiment == 'cellmambav2':
        model = CellMambaV2(num_classes=num_classes)

    elif config.experiment == 'cellmambav3':
        model = CellMambaV3(num_classes=num_classes)

    return model
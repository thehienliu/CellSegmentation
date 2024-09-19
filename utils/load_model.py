from models.cellmamba import CellMamba
from models.cellmamba2 import CellMambaV2
#from models.cellmamba3 import CellMambaV3
from models.cellvmamba import CellVMamba
from models.cellvmamba2 import CellVMamba2 

def load_model(config, num_classes):
    if config.experiment == 'cellvmambatiny':
        model = CellVMamba(num_classes=num_classes,
                  num_nuclei_classes=6,
                  pretrained_weight=config.weight_path)

    elif config.experiment == 'cellvmambabase':
        model = CellVMamba(num_classes=num_classes,
                  num_nuclei_classes=6,
                  dims=[128, 256, 512, 1024],
                  depths=[2, 2, 15, 2],
                  ssm_ratio=2.0,
                  drop_path_rate=0.6,
                  pretrained_weight=config.weight_path)
        
    elif config.experiment == 'cellvmambav2':
        model = CellVMamba2(
            num_classes=num_classes, 
            num_nuclei_classes=6,
            pretrained_weight=config.weight_path)
    
    elif config.experiment == 'cellvmambav2base':
        model = CellVMamba2(num_classes=num_classes,
                  num_nuclei_classes=6,
                  dims=[128, 256, 512, 1024],
                  depths=[2, 2, 15, 2],
                  ssm_ratio=2.0,
                  drop_path_rate=0.6,
                  pretrained_weight=config.weight_path)

    return model
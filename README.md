# Mamba Architecture for Cell Segmentation Task

This repository contains the implementation of the Mamba architecture for the Cell Segmentation task. Inspired by VMamba, this new architecture utilizes a prebuilt Mamba Block and different scanning methods instead of the VSS Block.
![CellMamba-v1](https://github.com/user-attachments/assets/d815284a-bfe0-4ce4-a554-28458c049a13)

## Installation
To set up the project, follow these steps:
```bash
git clone https://github.com/thehienliu/MambaForCellSegmentation.git
cd MambaForCellSegmentation
pip install -r requirements.txt
```

## Datasets
The primary dataset used in this research is the PanNuke dataset, provided by Jevgenij Gamper and colleagues. You can find more information about the dataset here: [PanNuke Dataset](https://arxiv.org/abs/2003.10778).

We preprocess the dataset using the code from the CellViT repository: [CellViT GitHub](https://github.com/TIO-IKIM/CellViT.git).

## Architecture
In this implementation, we replace the Vision Transformers encoder with our custom Mamba encoder. The architecture includes:

* CellMamba-v1: Uses the CellViT decoder.
* CellMamba-v2: Uses the Vision Mamba UNet decoder.
* CellMamba-v3: Uses the Mamba Decoder with ViT Encoder to evaluate potential improvements.


## Usage
To train the model with the provided configuration, use the following command:

```bash
python run.py --config config/cellmamba-v1.yaml
```

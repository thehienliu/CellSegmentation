import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from scipy.ndimage import center_of_mass
from utils.tools import get_bounding_box

class CustomCellSeg(Dataset):
    # AKA ChatGPT suggest the __init__ function
    def __init__(self, image_dir, label_dir, class_names, transforms=None):
        # Create dictionaries to map filenames (without extension) to full paths
        self.image_paths = {}
        self.label_paths = {}

        # Fill the dictionaries with the file paths
        for image_path in glob.glob(os.path.join(image_dir, '*.png')):
            filename = os.path.splitext(os.path.basename(image_path))[0]
            self.image_paths[filename] = image_path

        for label_path in glob.glob(os.path.join(label_dir, '*.npy')):
            filename = os.path.splitext(os.path.basename(label_path))[0]
            self.label_paths[filename] = label_path

        # Find matching filenames between images and labels
        matching_filenames = set(self.image_paths.keys()).intersection(self.label_paths.keys())

        # Store the matched file paths
        self.image_files = [self.image_paths[filename] for filename in matching_filenames]
        self.label_files = [self.label_paths[filename] for filename in matching_filenames]

        self.tissue_types = class_names
        self.transforms = transforms

    def __len__(self):
        # Return the number of matching image-label pairs
        return len(self.image_files)

    def __getitem__(self, idx):

        # Load image and label based on the matched file paths
        image_path = self.image_files[idx]
        label_path = self.label_files[idx]

        image = np.array(Image.open(image_path))
        mask = np.load(label_path)

        # Apply transform to mask and image
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Get transformed image and mask
        tissue_type = self.tissue_types[label_path.split('type_')[-1].split('.')[0]]
        type_map = mask[..., 0]
        inst_map = mask[..., 1]
        np_map = inst_map.copy()
        np_map[np_map > 0] = 1
        hv_map = CustomCellSeg.gen_instance_hv_map(inst_map)

        # Torch convert
        image = torch.Tensor(image).type(torch.float32)
        image = image.permute(2, 0, 1)
        if torch.max(image) >= 5:
            image = image / 255

        masks = {
            "instance_map": torch.Tensor(inst_map).type(torch.int64),
            "nuclei_type_map": torch.Tensor(type_map).type(torch.int64),
            "nuclei_binary_map": torch.Tensor(np_map).type(torch.int64),
            "hv_map": torch.Tensor(hv_map).type(torch.float32),
        }

        return image, masks, tissue_type

    @staticmethod
    def gen_instance_hv_map(inst_map: np.ndarray) -> np.ndarray:
        """Obtain the horizontal and vertical distance maps for each
        nuclear instance.

        Args:
            inst_map (np.ndarray): Instance map with each instance labelled as a unique integer
                Shape: (H, W)
        Returns:
            np.ndarray: Horizontal and vertical instance map.
                Shape: (2, H, W). First dimension is horizontal (horizontal gradient (-1 to 1)),
                last is vertical (vertical gradient (-1 to 1))
        """
        orig_inst_map = inst_map.copy()  # instance ID map

        x_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)
        y_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)

        inst_list = list(np.unique(orig_inst_map))
        inst_list.remove(0)  # 0 is background
        for inst_id in inst_list:
            inst_map = np.array(orig_inst_map == inst_id, np.uint8)
            inst_box = get_bounding_box(inst_map)

            # expand the box by 2px
            # Because we first pad the ann at line 207, the bboxes
            # will remain valid after expansion
            if inst_box[0] >= 2:
                inst_box[0] -= 2
            if inst_box[2] >= 2:
                inst_box[2] -= 2
            if inst_box[1] <= orig_inst_map.shape[0] - 2:
                inst_box[1] += 2
            if inst_box[3] <= orig_inst_map.shape[0] - 2:
                inst_box[3] += 2

            # improvement
            inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

            if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
                continue

            # instance center of mass, rounded to nearest pixel
            inst_com = list(center_of_mass(inst_map))

            inst_com[0] = int(inst_com[0] + 0.5)
            inst_com[1] = int(inst_com[1] + 0.5)

            inst_x_range = np.arange(1, inst_map.shape[1] + 1)
            inst_y_range = np.arange(1, inst_map.shape[0] + 1)
            # shifting center of pixels grid to instance center of mass
            inst_x_range -= inst_com[1]
            inst_y_range -= inst_com[0]

            inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

            # remove coord outside of instance
            inst_x[inst_map == 0] = 0
            inst_y[inst_map == 0] = 0
            inst_x = inst_x.astype("float32")
            inst_y = inst_y.astype("float32")

            # normalize min into -1 scale
            if np.min(inst_x) < 0:
                inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
            if np.min(inst_y) < 0:
                inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
            # normalize max into +1 scale
            if np.max(inst_x) > 0:
                inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
            if np.max(inst_y) > 0:
                inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

            ####
            x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            x_map_box[inst_map > 0] = inst_x[inst_map > 0]

            y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            y_map_box[inst_map > 0] = inst_y[inst_map > 0]

        hv_map = np.stack([x_map, y_map])
        return hv_map
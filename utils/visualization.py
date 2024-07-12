import torch
import matplotlib.pyplot as plt
from skimage.color import label2rgb

def plot_pred(outputs, masks, map):
    
    bs = outputs['nuclei_type_map'].shape[0]

    fig, ax = plt.subplots(2, bs, figsize=(12, 5))
    ax = ax.flatten()

    if map == 'nuclei_type_map':
        for i in range(0, len(ax)//2):
            ax[i].imshow(label2rgb(outputs['nuclei_type_map'].argmax(dim=1)[i].cpu().numpy(), bg_label=0))
            ax[i+bs].imshow(label2rgb(masks['nuclei_type_map'][i].numpy(), bg_label=0))

    if map == 'hv_map':
        for i in range(0, len(ax)//2):
            ax[i].imshow(outputs['hv_map'][:, 0, :, :][i].cpu().numpy())
            ax[i+bs].imshow(masks['hv_map'][:, 0, :, :][i].numpy())
    
    if map == 'nuclei_binary_map':
        for i in range(0, len(ax)//2):
            ax[i].imshow(outputs['nuclei_binary_map'].argmax(dim=1)[i].cpu().numpy(), cmap='gray')
            ax[i+bs].imshow(masks['nuclei_binary_map'][i].numpy(), cmap='gray')
    
    
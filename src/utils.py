import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def mask_to_rgb(mask, ignore_index=-1):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    colors = {
        0: [128, 128, 128],
        1: [255, 0, 0],
        2: [0, 255, 0],
        3: [0, 0, 255],
        4: [255, 255, 0],
        ignore_index: [0, 0, 0]
    }

    for class_id, color in colors.items():
        rgb[mask == class_id] = color

    return rgb


def visualize_sample(image, mask, prediction=None, title="Sample", figsize=(15, 5)):
    if prediction is not None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes = [axes[0], axes[1]]

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(mask_to_rgb(mask))
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    if prediction is not None:
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().numpy()
        axes[2].imshow(mask_to_rgb(prediction))
        axes[2].set_title('Prediction')
        axes[2].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Supervised Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].axis('off')

    axes[1, 0].plot(history['val_miou'], label='Val mIoU')
    axes[1, 0].set_title('Mean IoU')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mIoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['val_acc'], label='Val Accuracy')
    axes[1, 1].set_title('Pixel Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint['metrics']

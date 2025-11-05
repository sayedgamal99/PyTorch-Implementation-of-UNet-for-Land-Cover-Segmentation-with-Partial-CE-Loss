import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import cv2
import re


def natural_sort_key(path):
    parts = re.split(r'(\d+)', str(path.name))
    return [int(part) if part.isdigit() else part for part in parts]


def split_dataset_into_patches(data_dir, target_size=512, force=False):
    data_dir = Path(data_dir).resolve()
    images_dir = data_dir / 'images'
    masks_dir = data_dir / 'masks'
    patches_dir = data_dir / 'patches'

    if patches_dir.exists() and not force:
        num_images = len(list(patches_dir.glob('*.[jJ][pP][gG]')))
        num_masks = len(list(patches_dir.glob('*_m.[pP][nN][gG]')))

        if num_images > 0 and num_masks > 0:
            print(
                f"✓ Patches already exist: {num_images} images, {num_masks} masks")
            print(f"  Location: {patches_dir}")
            return patches_dir

    patches_dir.mkdir(exist_ok=True)
    print(f"\n📦 Splitting tiles into {target_size}x{target_size} patches...")
    print(f"  Source: {images_dir}")
    print(f"  Output: {patches_dir}")

    img_paths = sorted(list(images_dir.glob('*.tif')))
    mask_paths = sorted(list(masks_dir.glob('*.tif')))

    if len(img_paths) == 0 or len(mask_paths) == 0:
        raise FileNotFoundError(
            f"No tiles found in {images_dir} or {masks_dir}")

    img_paths = sorted(img_paths, key=natural_sort_key)
    mask_paths = sorted(mask_paths, key=natural_sort_key)

    print(f"  Found {len(img_paths)} tiles to split")

    total_patches = 0

    for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths), 1):
        img_filename = img_path.stem
        mask_filename = mask_path.stem

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path))

        if img is None or mask is None:
            print(f"  ⚠ Warning: Could not read {img_filename}, skipping")
            continue

        assert img_filename == mask_filename and img.shape[:2] == mask.shape[:2], \
            f"Mismatch: {img_filename} vs {mask_filename}"

        k = 0
        tile_patches = 0

        for y in range(0, img.shape[0], target_size):
            for x in range(0, img.shape[1], target_size):
                img_tile = img[y:y + target_size, x:x + target_size]
                mask_tile = mask[y:y + target_size, x:x + target_size]

                if img_tile.shape[0] == target_size and img_tile.shape[1] == target_size:
                    out_img_path = patches_dir / f"{img_filename}_{k}.jpg"
                    out_mask_path = patches_dir / f"{mask_filename}_{k}_m.png"

                    cv2.imwrite(str(out_img_path), img_tile)
                    cv2.imwrite(str(out_mask_path), mask_tile)

                    tile_patches += 1
                    total_patches += 1

                k += 1

        if i % 10 == 0 or i == len(img_paths):
            print(
                f"  Progress: {i}/{len(img_paths)} tiles processed, {total_patches} patches created")

    print(f"\n✓ Splitting complete!")
    print(f"  Total patches created: {total_patches}")
    print(f"  Location: {patches_dir}")

    return patches_dir


def download_landcover_dataset(data_dir='data', kaggle_json_path=None):
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    image_dir = data_dir / 'images'
    mask_dir = data_dir / 'masks'

    # Check if dataset already exists
    if image_dir.exists() and mask_dir.exists():
        num_images = len(list(image_dir.glob('*.tif')))
        num_masks = len(list(mask_dir.glob('*.tif')))

        if num_images > 0 and num_masks > 0:
            print(
                f"✓ Dataset already exists: {num_images} images, {num_masks} masks")
            print(f"  Location: {data_dir}")
            return data_dir

    if kaggle_json_path is None:
        kaggle_json_path = Path(__file__).parent.parent

    os.environ['KAGGLE_CONFIG_DIR'] = str(kaggle_json_path)

    api = KaggleApi()
    api.authenticate()

    kaggle_slug = 'adrianboguszewski/landcoverai'
    print(f"Downloading Kaggle dataset: {kaggle_slug} → {data_dir}")
    api.dataset_download_files(kaggle_slug, path=str(data_dir), unzip=True)

    if image_dir.exists() and mask_dir.exists():
        num_images = len(list(image_dir.glob('*.tif')))
        num_masks = len(list(mask_dir.glob('*.tif')))
        print(f"✓ Download complete: {num_images} images, {num_masks} masks")
    else:
        print("⚠ Warning: Expected 'images/' and 'masks/' directories not found")

    return data_dir


def mask_labels_random(mask, labeled_fraction, seed=42):
    rng = np.random.RandomState(seed)
    h, w = mask.shape
    total_pixels = h * w
    num_labeled = int(total_pixels * labeled_fraction)

    flat_mask = mask.astype(np.int32).flatten().copy()
    indices = rng.permutation(total_pixels)
    unlabeled_indices = indices[num_labeled:]
    flat_mask[unlabeled_indices] = -1

    return flat_mask.reshape(h, w).astype(np.int32)


class LandCoverDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split='train',
        labeled_fraction=1.0,
        transform=None,
        seed=42,
        use_split_file=True
    ):
        self.data_dir = Path(data_dir)
        self.patches_dir = self.data_dir / 'patches'
        self.labeled_fraction = labeled_fraction
        self.transform = transform
        self.seed = seed
        self.use_split_file = use_split_file

        if not self.patches_dir.exists():
            raise FileNotFoundError(
                f"Patches directory not found: {self.patches_dir}\n"
                f"Please run download_landcover_dataset() first or call split_dataset_into_patches()"
            )

        if use_split_file:
            split_file = self.data_dir / f'{split}.txt'
            if not split_file.exists():
                raise FileNotFoundError(f"Split file not found: {split_file}")

            with open(split_file, 'r') as f:
                self.patch_ids = [line.strip() for line in f if line.strip()]

            print(f"✓ Loaded {len(self.patch_ids)} patches from {split}.txt")
        else:
            # When not using split file, load all patches
            patch_images = sorted(
                list(self.patches_dir.glob('*.[jJ][pP][gG]')),
                key=natural_sort_key)
            # Extract patch IDs (remove .jpg extension)
            self.patch_ids = [p.stem for p in patch_images]
            print(
                f"✓ Loaded {len(self.patch_ids)} patches from {self.patches_dir}")

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]

        # Load pre-split patches
        img_path = self.patches_dir / f'{patch_id}.jpg'
        mask_path = self.patches_dir / f'{patch_id}_m.png'

        if not img_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Patch not found: {patch_id}")

        # Load image and mask
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path))

        # Handle multi-channel masks
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # Convert to int32 to handle -1 properly
        mask = mask.astype(np.int32)

        # Apply partial label masking if needed
        if self.labeled_fraction < 1.0:
            mask = mask_labels_random(
                mask, self.labeled_fraction, seed=self.seed + idx)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            mask = mask.long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask


def get_train_transform(resize_to=None):
    """
    Get training transforms with optional resizing.

    Args:
        resize_to (int, optional): If provided, resize patches to this size.
                                   Examples: 256, 384, 512
                                   None = keep original size (512x512)
    """
    transforms = []

    # Add resize if specified
    if resize_to is not None:
        transforms.append(A.Resize(resize_to, resize_to))

    # Standard augmentations
    transforms.extend([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return A.Compose(transforms)


def get_val_transform(resize_to=None):
    """
    Get validation transforms with optional resizing.

    Args:
        resize_to (int, optional): If provided, resize patches to this size.
                                   Examples: 256, 384, 512
                                   None = keep original size (512x512)
    """
    transforms = []

    # Add resize if specified
    if resize_to is not None:
        transforms.append(A.Resize(resize_to, resize_to))

    # Standard normalization
    transforms.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return A.Compose(transforms)

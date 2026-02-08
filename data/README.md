# Data Directory

This folder contains the **LandCover.ai dataset** and supporting visualization assets.

## 📂 Directory Structure

```
data/
├── images/                      # 41 aerial orthophotos (9636×9095 px each)
├── masks/                       # Corresponding segmentation masks (5-class labels)
├── patches/                     # Extracted 512×512 patches (generated during data prep)
├── samples/                     # Sample images for quick visualization
│
├── train.txt                    # 7,470 training patch IDs
├── val.txt                      # 1,602 validation patch IDs  
├── test.txt                     # 1,603 test patch IDs
│
├── unet.png                     # UNet architecture diagram
├── unet++.png                   # UNet++ architecture diagram
├── partial_labels_demo.png      # Visualization of partial supervision concept
└── sample_visualization.png     # Dataset sample with masks
```

---

## 🖼️ Visualization Assets

This folder contains several PNG files used in the README and documentation:

### Architecture Diagrams

<details>
<summary><b>Click to view architecture diagrams</b></summary>

#### `unet.png` - UNet Architecture
<img src="unet.png" alt="UNet Architecture" width="70%" />

Classic encoder-decoder with symmetric skip connections.

#### `unet++.png` - UNet++ Architecture  
<img src="unet++.png" alt="UNet++ Architecture" width="70%" />

Nested U-Net with dense skip pathways for improved feature fusion.

</details>

### Dataset Visualizations

- **`partial_labels_demo.png`**: Demonstrates how partial supervision works by showing the same image with different label fractions (100%, 70%, 50%, 30%)
- **`sample_visualization.png`**: Example satellite images paired with their segmentation masks from the LandCover.ai dataset

---

## 📥 Dataset Setup

The actual image and mask files are **not tracked by git** due to size.

**To download the dataset**, run:
```bash
jupyter notebook notebooks/01_data_exploration_landcover.ipynb
```

Or download the pre-processed data from [Google Drive](https://drive.google.com/drive/folders/1bSWFqZ1xEKoa3_PWlK2Nu3Q2TyJLQpvt?usp=sharing).

---

## 📊 Dataset Specifications

- **Source**: [LandCover.ai](https://landcover.ai/)
- **Format**: RGB orthophotos + segmentation masks (PNG)
- **Original Size**: 9636×9095 pixels per tile
- **Patch Size**: 512×512 pixels (resized to 384×384 during training)
- **Classes**: 5 (background, buildings, woodlands, water, roads)
- **Train/Val/Test Split**: 7,470 / 1,602 / 1,603 patches

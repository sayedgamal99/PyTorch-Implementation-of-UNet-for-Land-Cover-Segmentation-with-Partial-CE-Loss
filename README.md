# A Pure PyTorch Implementation of UNets for Land Cover Segmentation with Partial Supervision

<img src="runs/comparison_plots.png" alt="Qualitative comparison of UNet / UNet++ predictions under partial supervision" width="100%" />

<p align="center" style="font-size:10px;"><b>If this helps your research or learning, please ⭐ the repo & share!</b></p>

## 🌍 Why This Project Matters

Traditional semantic segmentation pipelines break down when a large portion of pixels are unlabeled. This repository shows how far you can push a model to still learn robust, spatially coherent land cover representations when much of the supervision is intentionally removed. Our custom **Partial Cross-Entropy Loss** only backpropagates through known pixels while letting the network internally infer missing regions by exploiting neighborhood context, multi-scale feature reuse, and structural priors in the data—without any extra "inpainting" modules or post-processing tricks.

In practice this means:

- You can mask out massive fractions of the annotation (e.g. majority of pixels) and the model still converges.
- The network learns to fill gaps implicitly by leveraging learned correlations (roads near buildings, water boundaries, woodland textures, etc.).
- No synthetic labels, no manual heuristics—just elegant partial supervision.

This highlights a core strength of deep learning: emergent relational understanding from sparse signals. The experiments here provide a clean, from-scratch PyTorch implementation to study that phenomenon in land cover mapping.

**Everything built from scratch in pure PyTorch - no pre-trained models, no external segmentation libraries**

## 🔥 Custom Implementations Showcase

### ✨ Custom Model Architecture

- **UNet & UNet++** built entirely from scratch
- No `segmentation-models-pytorch`, no `torchvision.models`
- Complete encoder-decoder with skip connections
- [📐 View architecture diagrams](#-model-architectures)

### ✨ Custom Loss Function

- **Partial Cross-Entropy Loss** for incomplete annotations
- Handles unlabeled pixels gracefully
- Novel approach to partial supervision

### ✨ Custom Data Handling

- **LandCoverDataset** with patch extraction
- Natural sorting algorithm for image-mask pairing
- Partial label simulation at pixel level

### ✨ Custom Training Pipeline

- Production-ready Trainer class
- Automatic checkpointing and metric tracking
- Complete validation infrastructure

---

## 📋 Project Highlights

- ✅ **100% Custom PyTorch Code** - No pre-trained weights, every line written from scratch
- ✅ **Partial Supervision** - Train with only 10-15% labeled pixels
- ✅ **LandCover.ai Dataset** - 41 orthophotos, 512×512 patches, 5 classes
- ✅ **Comprehensive Metrics** - IoU, pixel accuracy with proper ignore handling
- ✅ **Reproducible** - Fixed seeds, split files, deterministic training


---

## 📐 Model Architectures

Both architectures implemented **from scratch** in pure PyTorch—no pre-trained weights, no external libraries.

<details>
<summary><b>📊 Click to view UNet & UNet++ architecture diagrams</b></summary>

### UNet Architecture

<img src="data/unet.png" alt="UNet Encoder-Decoder Architecture" width="70%" />

**UNet** uses a symmetric encoder-decoder structure with skip connections that directly concatenate features from corresponding encoder levels to decoder levels, enabling precise localization.

### UNet++ Architecture

<img src="data/unet++.png" alt="UNet++ Nested Architecture" width="70%" />

**UNet++** introduces nested and dense skip pathways, creating multiple upsampling paths at different semantic levels. This reduces the semantic gap between encoder and decoder features, potentially improving segmentation accuracy.

</details>

---

## 🗂️ Project Structure

```
pure-pytorch-unet-landcover/
│
├── src/                          # Custom implementations (all from scratch)
│   ├── models.py                 # ✨ UNet & UNet++ architectures
│   ├── losses.py                 # ✨ Partial Cross-Entropy Loss
│   ├── dataset.py                # ✨ Custom dataset with patch extraction
│   ├── metrics.py                # ✨ IoU & accuracy metrics
│   ├── train.py                  # ✨ Training pipeline
│   └── utils.py                  # Visualization & helpers
│
├── notebooks/                    # Step-by-step execution
│   ├── 01_data_exploration_landcover.ipynb
│   ├── 02_core_implementation.ipynb
│   └── 03_experiments_training.ipynb
│
├── data/                         # LandCover.ai dataset (downloaded)
├── runs/                         # Training outputs & checkpoints
│
├── technical_report.md           # Detailed methodology & results
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/sayedgamal99/PyTorch-Implementation-of-UNet-for-Land-Cover-Segmentation-with-Partial-CE-Loss.git "torch-unet-partialCE"
cd "torch-unet-partialCE"

# Install dependencies
pip install -r requirements.txt

# OR use conda
conda env create -f environment.yml
conda activate landcover-seg
```

### 2. Dataset Setup

**Option 1: Download from Kaggle**

1. Get your Kaggle API credentials from https://www.kaggle.com/settings
2. Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `%USERPROFILE%\.kaggle\` (Windows)
3. Run notebook 01 to download the dataset

**Option 2: Use Pre-processed Data & Trained Models**

If you want the trained models and the ready-to-use cropped data, I uploaded both on this link:

- **[Google Drive Link](https://drive.google.com/drive/folders/1bSWFqZ1xEKoa3_PWlK2Nu3Q2TyJLQpvt?usp=sharing)**
- Extract to project root (creates `data/` and `runs/` folders)

### 3. Run Notebooks

Execute notebooks in order:

```bash
# 1. Download dataset and explore
jupyter notebook notebooks/01_data_exploration_landcover.ipynb

# 2. Test all modules
jupyter notebook notebooks/02_core_implementation.ipynb

# 3. Run full experiments (3-6 hours with GPU)
jupyter notebook notebooks/03_experiments_training.ipynb
```

---

## 📊 Dataset: LandCover.ai

- **Source**: [LandCover.ai](https://landcover.ai/)
- **Size**: 41 orthophotos (9636×9095 pixels each)
- **Patches**: 512×512 pixels, extracted on-the-fly
- **Splits**: 7,470 train / 1,602 val / 1,603 test
- **Classes**: 5 (background, buildings, woodlands, water, roads)

### Data Structure

```
data/
├── images/              # 41 large tiles
├── masks/               # Corresponding masks
├── train.txt            # 7,470 patch IDs
├── val.txt              # 1,602 patch IDs
└── test.txt             # 1,603 patch IDs
```

---

## 🎯 Understanding Partial-Supervision

<details>
<summary><b>What does "Partial-Supervision" mean in practice?</b></summary>

In traditional semantic segmentation, **every pixel** in the training images has a ground-truth label. **Partial-supervision** means we intentionally mask out a large percentage of these labels during training:

- **10% supervision**: Only 10% of pixels have known labels, 90% are marked as "unlabeled" (-1)
- **15% supervision**: Only 15% of pixels have known labels, 85% are marked as "unlabeled" (-1)
- The model must learn to **infer missing labels from spatial context**

### Visual Example

<img src="data/partial_labels_demo.png" alt="Partial supervision visualization" width="100%" />

The visualization above shows the same image with different label fractions. Notice how the masks become progressively sparser—yet our models can still learn robust segmentation from these sparse signals!

### How It Works

1. **Random pixel-level masking**: During training, we randomly select which pixels to keep labeled
2. **Partial Cross-Entropy Loss**: Only computes loss on labeled pixels, ignoring the rest
3. **Spatial learning**: The network learns to fill gaps by exploiting correlations (e.g., roads are linear, water bodies are smooth)

</details>

---

## 🧪 Experiments

### Training Configuration

```python
Label Fractions: 10%, 15%
Architectures: UNet, UNet++ (5 output classes)
Optimizer: Adam (lr=1e-5, weight_decay=1e-4)
Batch Size: 32
Epochs: 12
Patch Size: 512×512 (resized to 384×384)
Mixed Precision: Yes (AMP)
```

### Expected Results

After running Notebook 3, you'll get:

- **Trained models** (`runs/{unet,unetplusplus}_frac{10,15}/best_model.pth`)
- **Training curves** (loss and mIoU plots)
- **Results summary** (metrics for all 4 experiments)
- **Qualitative predictions** (visual comparison of predictions vs ground truth)

---

## 📊 Training Results

### Training History Across All Experiments

The plots below show training/validation loss, validation mIoU, and pixel accuracy for all 4 experiments (UNet/UNet++ × 10%/15% labeled pixels):

<img src="runs/training_curves_all.png" alt="Training curves for all experiments" width="100%" />

**Key Observations:**
- The Networks still able to learn even on this tough conditions.

### Qualitative Predictions

<details>
<summary><b>📸 Click to view detailed prediction visualizations</b></summary>

<img src="runs/predictions.png" alt="Model predictions vs ground truth on test samples" width="100%" />

**What you're seeing:**
- **Column 1**: Original satellite image
- **Column 2**: Ground truth segmentation mask (fully labeled)
- **Column 3**: Partial labels used during training (only 10% of pixels)
- **Column 4**: Model prediction (UNet++ trained on only 10% labels!)

Notice how the model successfully recovers the **road structure** in the top row even though only sparse pixels were labeled during training. This demonstrates the power of partial-supervision!

</details>

---

## 🔧 Technical Details

### Partial Cross-Entropy Loss

The core innovation enabling sparse supervision:


$$L_{\text{partial}} = -\frac{1}{|V|} \sum_{i \in V} \log P(y_i \mid x_i)$$

**Where:**

* **(V)** — Set of pixels with known labels (labels not equal to `-1`)
* **(|V|)** — Number of labeled pixels
* **(y_i)** — Ground truth class at pixel *i*
* **(P(y_i \mid x_i))** — Predicted probability of the true class at pixel *i*


**Key Properties:**
- Only computes loss on **labeled pixels** (ignore_index=-1)
- Normalizes by number of **labeled** pixels (not total pixels)
- Gracefully handles batches with very few labeled pixels

### Partial Label Simulation

During training, we simulate partial supervision through random pixel-level masking:

1. **Original mask**: All pixels labeled with classes 0-4 (background, buildings, woodlands, water, roads)
2. **Partial mask**: Random fraction (e.g., 90%) set to -1 ("unlabeled")
3. **Training**: Model only receives gradients from the remaining 10% labeled pixels
4. **Inference**: Model predicts all pixels (100% coverage)

### Model Architecture

**UNet**:

- Encoder: 5 levels, 64 base channels
- Decoder: skip connections, upsampling
- Output: 5 channels (one per class)

---

## 📈 Usage Example

```python
from src import (
    LandCoverDataset, get_train_transform,
    get_unet, PartialCrossEntropyLoss, Trainer
)
import torch

# Load dataset with 10% labeled pixels
train_dataset = LandCoverDataset(
    data_dir='data',
    split='train',
    transform=get_train_transform(resize_to=384),
    labeled_fraction=0.1,  # 10% labeled pixels
    use_split_file=True
)

# Create model and loss
model = get_unet(model_type='unet', classes=5, in_channels=3)
criterion = PartialCrossEntropyLoss(ignore_index=-1, weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

# Train
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device='cuda',
    num_classes=5,
    use_amp=True  # Mixed precision training
)
history = trainer.fit(num_epochs=12)
```

---

## 🛠️ Troubleshooting

### Out of Memory

```python
# Reduce batch size in notebook 3
batch_size = 2  # default is 4
```

### Dataset Not Found

Ensure Kaggle credentials are properly set and dataset downloads to `data/` directory.

### Slow Training

Use GPU for 10-20x speedup. Training on CPU not recommended for full experiments.

---

## 👤 Author

**By me [sayedgamal99](https://github.com/sayedgamal99)**

---

## 📚 References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In MICCAI 2015. https://arxiv.org/abs/1505.04597
- Zhou, Z., Siddiquee, M.M.R., Tajbakhsh, N., & Liang, J. (2018). UNet++: A Nested U-Net Architecture for Medical Image Segmentation. https://arxiv.org/abs/1807.10165
- LandCover.ai Dataset. https://landcover.ai/

---

**Date:** November 2025

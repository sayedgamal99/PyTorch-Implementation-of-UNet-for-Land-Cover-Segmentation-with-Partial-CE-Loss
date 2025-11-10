# A Pure PyTorch Implementation of UNet for Land Cover Segmentation with Partial Cross-Entropy Loss

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
- ✅ **Partial Supervision** - Train with 30%, 50%, 70% labeled pixels
- ✅ **LandCover.ai Dataset** - 41 orthophotos, 512×512 patches, 5 classes
- ✅ **Comprehensive Metrics** - IoU, pixel accuracy with proper ignore handling
- ✅ **Reproducible** - Fixed seeds, split files, deterministic training

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

## 🔧 Technical Details

### Partial Cross Entropy Loss

```python
L_partial = -1/|V| * Σ(i∈V) log P(y_i | x_i)
where V = {pixels with known labels (not -1)}
```

Only computes loss on labeled pixels, gracefully handles fully unlabeled batches.

### Partial Label Simulation

Random pixel-level masking:

- Original mask: all pixels labeled (0-4)
- Partial mask: random fraction set to -1 (unlabeled)
- Model trained only on labeled pixels

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

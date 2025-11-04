# A Pure PyTorch Implementation of UNet for Land Cover Segmentation with Partial Cross-Entropy Loss

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
git clone https://github.com/YOUR_USERNAME/pure-pytorch-unet-landcover.git
cd pure-pytorch-unet-landcover

# Install dependencies
pip install -r requirements.txt

# OR use conda
conda env create -f environment.yml
conda activate landcover-seg
```

### 2. Dataset Setup

1. Get your Kaggle API credentials from https://www.kaggle.com/settings
2. Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `%USERPROFILE%\.kaggle\` (Windows)
3. Run notebook 01 to download the dataset

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
Label Fractions: 30%, 50%, 70%
Architecture: UNet (5 output classes)
Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
Batch Size: 4
Epochs: 30
Patch Size: 512×512
```

### Expected Results

After running Notebook 3, you'll get:

- **Trained models** (`runs/frac{30,50,70}_partial_ce/best_model.pth`)
- **Training curves** (loss and IoU plots)
- **Results table** (CSV with metrics for all experiments)
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

````python
from src import (
    LandCoverDataset, get_train_transform,
    get_unet, PartialCrossEntropyLoss, Trainer
)
import torch

# Load dataset with 30% labeled pixels
train_dataset = LandCoverDataset(
    data_dir='data',
    split='train',
    transform=get_train_transform(),
    labeled_fraction=0.3,
    use_split_file=True
)

# Create model and loss
**LandCoverDataset** - custom dataset class:
```python
dataset = LandCoverDataset(
    data_dir='data',
    split='train',
    transform=get_train_transform(),
    labeled_fraction=0.5,  # Simulate partial labels
    use_split_file=True
)
````

**Features:**

- On-the-fly patch extraction from 9636×9095 tiles
- Natural sorting for correct pairing
- Partial label simulation at pixel level
- No pre-processing required
- Memory-efficient streaming

### 4. Custom Training Pipeline

criterion = PartialCrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train

trainer = Trainer(
model=model,
train_loader=train_loader,
val_loader=val_loader,
criterion=criterion,
optimizer=optimizer,
device='cuda',
num_classes=5
)
history = trainer.fit(num_epochs=30)

````

---

## 🛠️ Troubleshooting

### Out of Memory

```python
# Reduce batch size in notebook 3
batch_size = 2  # default is 4
````

### Dataset Not Found

Ensure Kaggle credentials are properly set and dataset downloads to `data/` directory.

### Slow Training

Use GPU for 10-20x speedup. Training on CPU not recommended for full experiments.

---

## 📝 Citation

If you use this code, please cite the LandCover.ai dataset:

```bibtex
@article{boguszewski2020landcoverai,
  title={LandCover.ai: Dataset for automatic mapping of buildings, woodlands and water from aerial imagery},
  author={Boguszewski, Adrian and Batori, Dominik and Ziemba-Jankowska, Natalia and Dziedzic, Tomasz and Zambrzycka, Anna},
  journal={arXiv preprint arXiv:2005.02264},
  year={2020}
}
```

---

## 📄 License

This project is provided for educational and research purposes.

---

## 🙏 Acknowledgments

- **LandCover.ai** dataset creators
- **PyTorch** and **Albumentations** communities

---

## 📧 Contact

For questions or issues, please open an issue in the repository.

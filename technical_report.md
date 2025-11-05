# PyTorch Implementation of UNet for Land Cover Segmentation with Partial Cross-Entropy Loss

## 1. Introduction

This project implements semantic segmentation under **partial supervision** — training models with only a fraction of pixels labeled. We compare UNet and UNet++ architectures on land cover segmentation using a custom Partial Cross-Entropy loss function.

---

## 2. Method

### 2.1 Partial Cross-Entropy Loss

A custom loss function designed for partial supervision scenarios where only some pixels have ground-truth labels.

**Mathematical Formulation:**

$$L_{partial} = -\frac{1}{|V|} \sum_{i \in V} \log \text{softmax}_{y_i}(z_i)$$

where:

- $V$ = set of pixels with known labels (not marked as `ignore_index = -1`)
- $|V|$ = number of valid labeled pixels
- $z_i$ = model logits at pixel $i$
- $y_i$ = ground truth class at pixel $i$

**Key Properties:**

- Computes loss **only** on labeled pixels
- Automatically ignores unlabeled pixels (`ignore_index = -1`)
- Normalizes by number of labeled pixels (not total pixels)
- Enables training with sparse annotations

### 2.2 Model Architectures

Both architectures implemented from scratch in pure PyTorch:

**UNet:**

- 5-level encoder-decoder with skip connections
- Base channels: 64 (doubles at each level)
- BatchNorm + ReLU activation
- Classic U-shaped architecture

**UNet++:**

- Nested decoder with dense skip connections
- Enhanced feature aggregation across scales
- More parameters than standard UNet
- Improved gradient flow

**Common Configuration:**

- Input: RGB images (384×384)
- Output: 5-class segmentation maps
- Classes: Background, Buildings, Woodlands, Water, Roads

---

## 3. Experimental Setup

### 3.1 Dataset

**LandCover.ai** - High-resolution aerial imagery dataset

- **Total images:** 41 orthophotos (9636×9095 pixels each)
- **Patches:**
  - Training: 7,470
  - Validation: 1,602
  - Test: 1,603
- **Patch size:** 384×384 pixels
- **Classes:** 5 (background, buildings, woodlands, water, roads)
- **Augmentation:** Horizontal flip, vertical flip, rotation, shift-scale-rotate

### 3.2 Training Configuration

| Parameter       | Value     |
| --------------- | --------- |
| Optimizer       | Adam      |
| Learning Rate   | 1e-5      |
| Weight Decay    | 1e-4      |
| Batch Size      | 32        |
| Epochs          | 12        |
| Mixed Precision | Yes (AMP) |
| Device          | CUDA      |
| Random Seed     | 42        |

### 3.3 Experiments

**4 experiments** testing the impact of:

- **Label fractions:** 10%, 15%
- **Architectures:** UNet, UNet++

All models evaluated on **fully labeled** validation set for fair comparison.

---

## 3. Results

### 3.1 Quantitative Results

**Table 1: Performance across different label fractions**

| Label Fraction | Final Val mIoU | Final Val Accuracy | Best Val mIoU | Training Time |
| -------------- | -------------- | ------------------ | ------------- | ------------- |

### UNet Results

| Label Fraction | Final mIoU | Best mIoU | Final Accuracy |
| -------------- | ---------- | --------- | -------------- |
| 10%            | 0.4775     | 0.4887    | 0.8640         |
| 15%            | 0.4764     | 0.4902    | 0.8605         |

### UNet++ Results

| Label Fraction | Final mIoU | Best mIoU | Final Accuracy |
| -------------- | ---------- | --------- | -------------- |
| 10%            | 0.5054     | 0.5054    | 0.8680         |
| 15%            | 0.4630     | 0.4705    | 0.8398         |

### 4.2 Key Observations

**Best Model:** UNet++ with 10% labels

- **mIoU:** 0.5054
- **Accuracy:** 86.80%
- **Training:** Converged in 12 epochs

**Architecture Comparison:**

- **UNet++ @ 10%:** 0.5054 mIoU ✓ **Best performance**
- **UNet @ 15%:** 0.4902 mIoU
- **UNet @ 10%:** 0.4887 mIoU
- **UNet++ @ 15%:** 0.4705 mIoU

**Surprising Finding:** More labels ≠ better performance

- UNet++ performed **worse** with 15% labels (0.4705) than 10% labels (0.5054)
- UNet showed slight improvement from 10% to 15% (0.4887 → 0.4902)
- This suggests potential overfitting or suboptimal convergence with 15% labels

### 4.3 Training Dynamics

All experiments trained for **12 epochs** with consistent behavior:

**Training Loss:**

- Steady decrease across all experiments
- UNet converges faster initially
- UNet++ shows more stable convergence
- Final training loss: ~0.042-0.044

**Validation Loss:**

- Best validation loss: UNet @ 15% (0.0360)
- Lowest doesn't guarantee best mIoU
- All models showed good generalization

**Convergence:**

- Most models reached peak performance by epoch 10-12
- No significant overfitting observed
- Mixed precision training (AMP) enabled faster training

### 4.4 Class Weights

Class imbalance addressed with computed weights:

```
[0.66, 1.27, 1.12, 1.03, 1.91]
```

Classes with fewer pixels (buildings, roads) received higher weights.

---

## 5. Analysis & Insights

### 5.1 Impact of Label Fraction

**Unexpected Result:** 10% vs 15% labels showed **minimal difference**

- UNet: 0.4887 (10%) vs 0.4902 (15%) — only +0.15% improvement
- UNet++: 0.5054 (10%) vs 0.4705 (15%) — **degraded by 6.9%**

**Interpretation:**

1. **10% may be sufficient** for this dataset with proper training
2. Random masking at 15% might create harder learning scenarios
3. UNet++ benefits more from sparse but informative labels
4. Model capacity vs label quantity trade-off

### 5.2 Architecture Comparison

**UNet++ Advantages:**

- Better feature aggregation through nested skip connections
- Achieved best overall performance (0.5054 mIoU)
- More robust to label sparsity (at 10%)

**UNet Advantages:**

- More consistent across label fractions
- 2x faster training (1.5 min/epoch vs 3 min/epoch)
- Simpler architecture, easier to debug

**Trade-off:** UNet++ offers ~3% better mIoU but costs 2x training time.

### 5.3 Practical Implications

**For 10% label budget:**

- Use UNet++ → 50.54% mIoU
- Good performance with minimal annotation effort

**For 15% label budget:**

- Use UNet → 49.02% mIoU
- More stable and predictable

**ROI Analysis:**

- 50% more labels (10%→15%) gives minimal benefit
- Focus on **label quality** over quantity
- Strategic labeling may beat random masking

---

## 6. Conclusions

### 6.1 Key Findings

1. **Label Efficiency:** Models can achieve **~50% mIoU with only 10% labeled pixels** using Partial CE Loss

2. **Architecture Matters:** UNet++ achieved best performance (50.54% mIoU) but UNet offers better speed/performance trade-off

3. **Diminishing Returns:** Increasing labels from 10% to 15% provided **minimal improvement** and even degraded UNet++ performance

4. **Partial CE Loss Works:** Successfully trained models with sparse supervision, validating the loss function design

### 6.2 Recommendations

**For practitioners with limited annotation budgets:**

1. ✅ **Use Partial Cross-Entropy Loss** for partial supervision scenarios
2. ✅ **Start with 10% random labels** as baseline — may be sufficient
3. ✅ **Try UNet++ first** if computational budget allows
4. ✅ **Apply class weights** to handle imbalanced datasets
5. ⚠️ **Don't assume more labels = better** — validate on your dataset

**Optimal Setup (based on results):**

- Architecture: UNet++
- Label fraction: 10%
- Expected mIoU: ~50%
- Training time: ~35 minutes (12 epochs)

### 6.3 Limitations

1. **Evaluation:** Only tested at 10% and 15% — wider range needed (5%, 20%, 30%, 50%)
2. **Label Pattern:** Random pixel masking may not reflect real annotation scenarios
3. **Dataset:** Single dataset (LandCover.ai) — generalization unclear
4. **Metrics:** mIoU may not capture all quality aspects (boundary accuracy, small objects)
5. **No Semi-Supervised Methods:** Pure supervised approach only

### 6.4 Future Work

**Immediate Next Steps:**

1. Test more label fractions (5%, 20%, 30%, 50%) to find optimal point
2. Implement strategic labeling (uncertainty-based, entropy-based)
3. Add consistency regularization for semi-supervised learning

**Advanced Extensions:**

1. Evaluate on additional datasets (ISPRS Potsdam, DeepGlobe Land Cover)
2. Test with real sparse annotations (scribbles, points, bounding boxes)
3. Implement pseudo-labeling for unlabeled regions
4. Add test set evaluation for final performance
5. Compare with active learning strategies

---

## 7. References

1. **Ronneberger, O., et al.** "U-Net: Convolutional Networks for Biomedical Image Segmentation." _MICCAI 2015._

2. **Zhou, Z., et al.** "UNet++: A Nested U-Net Architecture for Medical Image Segmentation." _DLMIA 2018._

3. **Boguszewski, A., et al.** "LandCover.ai: Dataset for Automatic Mapping of Buildings, Woodlands and Water from Aerial Imagery." _arXiv:2005.02264, 2020._

---

## Appendix: Reproducibility

**Environment:**

- PyTorch 2.x with CUDA support
- Mixed Precision Training (AMP)
- Random seed: 42

**Training Time per Experiment:**

- UNet: ~18 minutes (12 epochs)
- UNet++: ~35 minutes (12 epochs)

**Saved Models:**

- `runs/unet_frac10/best_model.pth` — UNet @ 10%
- `runs/unet_frac15/best_model.pth` — UNet @ 15%
- `runs/unetplusplus_frac10/best_model.pth` — **UNet++ @ 10% (BEST)**
- `runs/unetplusplus_frac15/best_model.pth` — UNet++ @ 15%

**Visualizations:**
All training curves and predictions saved in `runs/` directory.

---

**Implementation:** Complete from-scratch PyTorch with no external segmentation libraries  
**Date:** November 2025

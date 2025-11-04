# A Pure PyTorch Implementation of UNet for Land Cover Segmentation with Partial Cross-Entropy Loss

## 1. Method

### 1.1 Partial Cross Entropy Loss

The Partial Cross Entropy Loss is designed to handle partial supervision scenarios where only a subset of pixels have valid ground-truth labels.

**Mathematical Formulation:**

$$L_{partial} = -\frac{1}{|V|} \sum_{i \in V} \log \text{softmax}_{y_i}(z_i)$$

where:

- $V$ = set of pixels with known labels (not equal to `ignore_index = -1`)
- $|V|$ = number of valid pixels
- $z_i$ = logits at pixel $i$
- $y_i$ = ground truth class at pixel $i$

**Key Properties:**

- Only computes loss on labeled pixels
- Ignores pixels marked with `ignore_index = -1`
- Normalizes by number of valid pixels, not total pixels
- Gradient flows only through labeled pixels
- Handles edge case of fully unlabeled batches gracefully

### 1.2 Model Architecture

**UNet Implementation (from scratch):**

- Encoder: 5-level downsampling path
- Base channels: 64 (doubles at each level)
- Decoder: 5-level upsampling path with skip connections
- Activation: ReLU
- Normalization: BatchNorm2d
- Output: 5 classes (background, buildings, woodlands, water, roads)

**UNet++ Implementation (from scratch):**

- Nested decoder architecture with dense skip connections
- Multiple upsampling paths
- Enhanced feature aggregation
- Same base configuration as UNet

---

## 2. Experiments

### 2.1 Purpose and Hypothesis

**Research Questions:**

1. How does the percentage of labeled pixels affect segmentation performance?
2. Can models trained with only 30-70% labeled pixels achieve reasonable performance?
3. How does partial supervision compare across different label fractions?

**Hypotheses:**

1. Performance will degrade gracefully as labeled pixel percentage decreases
2. The performance gap between 30% and 70% labeling will demonstrate the value of additional labels
3. Even with 30% labeling, the model should learn meaningful segmentation patterns

### 2.2 Experimental Setup

**Dataset:** LandCover.ai

- 5 classes: background (0), buildings (1), woodlands (2), water (3), roads (4)
- 41 high-resolution orthophotos (9636×9095 pixels each)
- Training patches: 7,470
- Validation patches: 1,602
- Test patches: 1,603
- Patch size: 512×512 pixels
- Data augmentation: horizontal flip, vertical flip, rotation, shift-scale-rotate

**Model Architecture:**

- UNet (from-scratch PyTorch implementation)
- Encoder: 5 levels with BatchNorm
- Decoder: Skip connections + upsampling
- Input: 3-channel RGB images (512×512)
- Output: 5-class segmentation maps (512×512)

**Training Configuration:**

- Optimizer: Adam
- Learning rate: 1e-3
- Weight decay: 1e-4
- Batch size: 4
- Epochs: 30
- Loss function: Partial Cross-Entropy (ignore_index=-1)
- Random seed: 42 (for reproducibility)

**Experimental Conditions:**

- Label fractions tested: 30%, 50%, 70%
- Training mode: Partial Cross-Entropy Loss on labeled pixels only
- Partial label simulation: Random pixel-level masking

---

## 3. Results

### 3.1 Quantitative Results

**Table 1: Performance across different label fractions**

| Label Fraction | Final Val mIoU | Final Val Accuracy | Best Val mIoU | Training Time |
| -------------- | -------------- | ------------------ | ------------- | ------------- |
| 30%            | [TBD]          | [TBD]              | [TBD]         | [TBD]         |
| 50%            | [TBD]          | [TBD]              | [TBD]         | [TBD]         |
| 70%            | [TBD]          | [TBD]              | [TBD]         | [TBD]         |

**Table 2: Per-class IoU (example: 50% labeling)**

| Class          | IoU   |
| -------------- | ----- |
| Background (0) | [TBD] |
| Buildings (1)  | [TBD] |
| Woodlands (2)  | [TBD] |
| Water (3)      | [TBD] |
| Roads (4)      | [TBD] |
| **Mean**       | [TBD] |

### 3.2 Training Curves

_[Training loss, validation loss, validation mIoU plots to be inserted]_

### 3.3 Qualitative Results

_[Sample predictions showing: original image, full ground truth, masked labels, predictions to be inserted]_

### 3.4 Analysis

**Impact of Label Fraction:**

- [To be filled after experiments - expected to show graceful degradation as label fraction decreases]

**Convergence Behavior:**

- [To be filled after experiments - training curves will show loss and mIoU progression]

**Class-Specific Performance:**

- [To be filled - analysis of which classes are easier/harder to segment with partial labels]

---

## 4. Conclusions

### 4.1 Key Findings

1. **Label Efficiency:** [To be filled]
2. **Consistency Benefits:** [To be filled]
3. **Practical Trade-offs:** [To be filled]

### 4.2 Recommendations

For practitioners working on remote sensing segmentation with limited annotations:

1. **Use Partial CE Loss:** Essential for partial supervision; never use standard CE with missing labels
2. **Target 50-70% labeling:** Likely provides good performance/cost trade-off (to be confirmed by results)
3. **Random masking simulation:** Useful for initial experiments before collecting real partial annotations
4. **Class balance matters:** Ensure all classes are represented in labeled pixels

### 4.3 Limitations

1. Synthetic partial labels via random masking may not reflect real annotation patterns
2. Single dataset evaluation (LandCover.ai)
3. Limited to UNet architecture (though UNet++ also implemented)
4. No advanced semi-supervised techniques (consistency regularization, pseudo-labeling, etc.)

### 4.4 Future Work

1. Implement consistency regularization for semi-supervised learning
2. Evaluate on additional remote sensing datasets (ISPRS Potsdam, DeepGlobe)
3. Explore active learning strategies for intelligent pixel selection
4. Test with real partial annotations (e.g., sparse point annotations, scribbles)
5. Add more advanced architectures (DeepLabV3+, HRNet)
6. Extend to multi-temporal and multi-spectral data

---

## 5. References

1. Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.
2. Zhou, Z., et al. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation." DLMIA 2018.
3. Boguszewski, A., et al. "LandCover.ai: Dataset for Automatic Mapping of Buildings, Woodlands and Water from Aerial Imagery." arXiv:2005.02264, 2020.

---

**Report Status:** Results to be filled after running experiments (Notebook 03)  
**Code Repository:** https://github.com/YOUR_USERNAME/pure-pytorch-unet-landcover  
**Implementation:** Complete from-scratch PyTorch implementation

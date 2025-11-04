# Data Directory

LandCover.ai dataset storage (not tracked by git).

## Structure

```
data/
├── images/          # 41 aerial orthophotos (9636×9095 px)
├── masks/           # Corresponding segmentation masks
├── train.txt        # 7,470 training patch IDs
├── val.txt          # 1,602 validation patch IDs
└── test.txt         # 1,603 test patch IDs
```

Run `01_data_exploration_landcover.ipynb` to download the dataset.

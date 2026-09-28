# Plant Disease Classification — CNN

Classifies PlantVillage leaf images into 38 crop/disease classes using a small
convolutional network in PyTorch.

## Setup

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pillow tqdm
```

Then place the dataset so that this path exists:

```
data/raw/color/<Class_Name>/*.JPG
```

The dataset is [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
(54,305 images, 38 classes). It is not in this repo — see `.gitignore`.

## Running

```bash
jupyter nbconvert --to notebook --execute Plant_Disease_CNN.ipynb
```

Or open it and run all cells top to bottom. The first pass decodes all 54,305
images to verify they are readable, which takes a few minutes.

Runs on CPU or CUDA automatically; batch size and data loading adapt to the
device. On CPU expect the run to be slow — reduce `MAX_EPOCHS` while iterating.

## Approach

| Stage | Choice |
|---|---|
| Split | 80/10/10 stratified, seed 42. Test set touched once at the end |
| Input | 128×128 RGB |
| Augmentation | Flip, ±15° rotation, colour jitter — **train set only** |
| Imbalance | Inverse-frequency `class_weight` in the loss (152 to 5,507 images/class) |
| Model | 4 conv blocks (32→64→128→192), each BatchNorm + ReLU + max-pool, then global average pool → dropout → linear |
| Optimiser | Adam, lr 1e-3, weight decay 1e-4 |
| Schedule | `ReduceLROnPlateau` on validation loss, halving to a 1e-5 floor |
| Early stop | Patience 3, best weights restored |
| Epochs | 15 |

## Evaluation

Reports accuracy, per-class precision/recall/F1, a confusion matrix, and
misclassified-image samples. The validation and test sets are never augmented or
reweighted, so the reported numbers reflect the natural class distribution.

## Layout

```
Plant_Disease_CNN.ipynb   all code and analysis
artifacts/                saved model weights
data/raw/color/           dataset (not committed)
```

## Notes

- Weights are git-ignored. To share them, publish as a release asset.
- The dataset is curated leaf imagery, so field generalisation is unverified.
- The class weighting helps accuracy on rare classes, but overall accuracy will
  read slightly lower than an unweighted run. That is the intended trade.

**Source:** Hughes, D. P. & Salathé, M. (2015), *An open access repository of
images on plant health to enable the development of mobile disease diagnostics*,
arXiv:1511.08060.

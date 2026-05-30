# sample-fruits

A small balanced image-classification subset for trying the image notebooks end-to-end on Colab's free tier without bringing your own data.

## Contents

| Class | Images |
|---|---|
| Banana | 30 |
| Mango | 30 |
| Pineapple | 30 |
| **Total** | **90** |

All images are 100×100 JPEG, ~7 KB each, total ~600 KB.

## Folder structure

```
sample-fruits/
├── Banana/
│   ├── 0_100.jpg
│   ├── 1_100.jpg
│   └── ...
├── Mango/
│   └── ...
└── Pineapple/
    └── ...
```

This is exactly the layout the IGNODE Custom Model Upload wizard accepts for image classification — one subfolder per class, images inside each subfolder.

## Attribution

These images are a subset of the **Fruits-360** dataset by Mihai Oltean, originally published on Kaggle under CC BY-SA 4.0. Source: https://www.kaggle.com/datasets/moltean/fruits.

This subset is redistributed under the same CC BY-SA 4.0 license. The full original dataset has ~131k images across 207 classes; this 90-image / 3-class slice is for demonstration purposes only — accuracy from training on it will be modest (60-80% range). For production-grade accuracy, bring a larger dataset of your own.

## Using this dataset

The image notebooks in this repo default to this sample so they run end-to-end out of the box. You'll see the notebook download these images from `raw.githubusercontent.com` automatically.

When you're ready to use your own data, edit the notebook's settings cell:

```python
SAMPLE_DATASET = None  # was 'sample-fruits' — None tells the notebook to upload your own .tar
```

Then drop your own folder of class subfolders (the browser tars it client-side and uploads as one `.tar`).

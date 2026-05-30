# IGNODE Colab Notebooks

Maintained Google Colab notebooks for training machine learning models that drop directly into the **IGNODE** platform via the Custom Model Upload wizard.

Use these when you want to train your own model with custom hyperparameters, more epochs, or your own backbone — and bring the resulting ONNX model back into IGNODE for inference.

## The notebooks

| Task | Notebook | Open in Colab |
|---|---|---|
| Image Classification | [`colab/train_image_classifier.ipynb`](colab/train_image_classifier.ipynb) | _(coming soon)_ |
| Image Detection | [`colab/train_image_detector.ipynb`](colab/train_image_detector.ipynb) | _(coming soon)_ |
| Tabular Classification | [`colab/train_tabular_classifier.ipynb`](colab/train_tabular_classifier.ipynb) | _(coming soon)_ |
| Tabular Regression | [`colab/train_tabular_regression.ipynb`](colab/train_tabular_regression.ipynb) | _(coming soon)_ |

Each notebook produces an `model.onnx` file plus the metadata sidecars the IGNODE upload wizard expects.

## How it works

1. Click an "Open in Colab" link above. The notebook opens in your own Google account.
2. (Image notebooks) Switch to GPU runtime — **Runtime → Change runtime type → GPU**.
3. Drop your dataset file (a `.tar` of class subfolders for image, a `.csv` for tabular) into the file panel on the left.
4. Run all cells (**Runtime → Run all**) and wait for training to complete.
5. The final cell downloads `model.onnx` to your laptop.
6. Open your IGNODE portal → **ML Factory → Custom Models → + Upload ML Model** and drop the file.

Your training dataset never touches IGNODE on its way to Colab. The trained model comes back to your laptop and you upload it yourself.

## Requirements

- A Google account (free tier of Colab is sufficient for the default datasets)
- Your training data on your laptop
- For image detection: a labeled dataset in COCO format (annotation tools are out of scope — see the notebook for tool suggestions)

## License

[MIT](LICENSE) — you can fork these and customize freely. PRs welcome.

---

Maintained by IGNODE. For questions and full guides, see the IGNODE documentation site.

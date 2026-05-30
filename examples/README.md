# Example Datasets

IoT-flavored sample datasets used by the Colab notebooks so they run end-to-end without you needing to upload anything first. Each notebook defaults to one of these — open it, click **Runtime → Run all**, and you'll have a trained model in a couple of minutes.

To use your own data instead, edit the notebook's settings cell:

```python
SAMPLE_DATASET = None     # was 'sensor_anomaly_classification' — None tells the notebook to prompt for your file
LABEL_COLUMN = 'YourLabel'  # whatever column you want to predict
```

## Datasets

| File | Task | Label column | What it predicts |
|---|---|---|---|
| `sensor_anomaly_classification.csv` | Classification | `Anomaly` | Sensor reading is `Normal` / `Warning` / `Critical` from temperature, humidity, vibration, pressure, RPM, power, sound, operating hours |
| `equipment_rul_regression.csv` | Regression | `RemainingLife` | Equipment remaining useful life (cycles) from cycle count, sensor readings, oil temp, coolant flow, power, ambient temp, load |
| `building_energy_regression.csv` | Regression | `HeatingLoad` | Building heating load (kWh) from wall area, roof area, height, orientation, glazing, compactness, surface area |

Each CSV has a header row, ~90 data rows, and one labeled target column. Small enough to download instantly from any Colab session via a single `pd.read_csv` against the raw GitHub URL.

## Notebook defaults

| Notebook | Defaults to |
|---|---|
| `train_tabular_classifier.ipynb` (advanced) | `sensor_anomaly_classification` |
| `train_tabular_regression.ipynb` (advanced) | `equipment_rul_regression` |
| `simple/train_tabular_classifier_*.ipynb` (3 of them) | `sensor_anomaly_classification` |
| `simple/train_tabular_regression_*.ipynb` (3 of them) | `equipment_rul_regression` |

## Bringing your own data

Once you've seen the notebook work end-to-end, swap to your own CSV:

1. In the settings cell, set `SAMPLE_DATASET = None`
2. Change `LABEL_COLUMN` to the column in your CSV you want to predict
3. Run all cells again — the notebook will prompt you to upload your CSV when it gets to the data-loading step

Everything else (training, evaluation, ONNX export, sidecar generation, IGNODE upload instructions) works the same regardless of which dataset you used.

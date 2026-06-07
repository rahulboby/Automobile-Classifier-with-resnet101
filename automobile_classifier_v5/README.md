# Automobile Classifier v5

Train and compare multiple image classifiers on the same automobile dataset split.

## Project layout

- `data/`: raw image folders, one folder per class.
- `data_split/`: generated train/validation/test folders.
- `prepare_data.py`: shared dataset splitting utility.
- `notebooks_models/`: model training notebooks.
- `notebooks_models/notebook_resnet50.ipynb`: ResNet-50 training notebook.
- `notebooks_models/notebook_resnet18.ipynb`: ResNet-18 training notebook.
- `notebooks_models/notebook_resnet101.ipynb`: ResNet-101 training notebook.
- `notebooks_models/notebook_densenet121.ipynb`: DenseNet-121 training notebook.
- `notebooks_models/notebook_efficientnet_b0.ipynb`: EfficientNet-B0 training notebook.
- `notebooks_models/notebook_mobilenet_v2.ipynb`: MobileNetV2 training notebook.

## Quick run-through

1. Install dependencies.

```powershell
pip install -r requirements.txt
```

2. Put your raw dataset in `data/`, with one folder per class.

```text
data/
    car/
    bike/
```

3. Create the shared train/validation/test split.

```powershell
python -m prepare_data
```

This reuses an existing `data_split/` folder if one is already present. To rebuild the split from scratch, run:

```powershell
python -m prepare_data --overwrite
```

4. Open one model notebook from `notebooks_models/` and run the cells from top to bottom.

Each notebook now uses the same prepared split and keeps the model-specific parts in the notebook: model loading, layer freezing, training, training curves, test metrics, confusion matrix, and final model save.

5. Find saved outputs.

- Checkpoints are saved in `checkpoints_<model_name>/`.
- Final models are saved in `models/final_model_<model_name>.pth`.
- Saved model files include `class_names`, `num_classes`, `image_size`, and `architecture`.

## Notes

- Use `utils/configs.py` to change image size, batch size, epoch count, learning rate, split ratios, and default paths.
- Run notebook cells from the project root so imports like `prepare_data` and `utils.configs` resolve cleanly.
- The notebook split cell calls `split_dataset(overwrite=False)`, so rerunning a notebook will not delete an existing split.
- Use `split_dataset(overwrite=True)` in a notebook only when you intentionally want a new split.

# Other Information:

### resnet18 
GPU Used - ~4.0/6.0 GB

Trained for 35 epochs (I think it is overfitted)
Did not perform well in manual single image test data in app.py - seltos back pic was classified as carnival with <30% confidence

### resnet50
GPU used - 4.6/6.0 GB

Planned on 35 epochs, but only 31 epochs done.

Stopped training at 32nd epoch due to overfitting - val acc decreased slightly while train acc increased
Therefore, using only 30 epochs as all the models start overfitting around 32-35 epochs

Performed well in manual single image test data in app.py - seltos back pic was classified correctly, but with ~35% confidence

### resnet101
GPU used - 4.6/6.0 GB

Planned for 30 epochs - stopped at 26 - started overfitting at 25
Back seltos pic predicted with 40% confidence - ***Best Yet***

### efficientnet_b0
GPU used - 5.6/6.0 GB
Shared GPU usage: 1.0/11.0 GB
Planned on 30 epochs, but auto cut of at 24 due to overfitting
Changes made: Added an automatic break in training loop if validation falls in a 3 epoch window - but there was a bug
ran for another 15 epochs by manually setting epoch number in loop

### densnet121
GPU used - 5.5 GB
Shared GPU usage: 0.8/11.0 GB
Ran 30 epochs, then ran another 10 - automatically stopped at 6 due to overfitting 
but ran another 10 epochs with ```patience``` set to 5 instead of 3 - improved continuously so ran for another 10
Total: 55 epochs
Predicted the back seltos picture 39% confidence
***Test results show that resnet101 is the best***







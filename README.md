# v0
This version automobile_classifier_v0 will classify the given photo into one of this:
- autorikshaw
- bike
- car
- fighterjet
- tank
- truck

This was a testing version to test out ResNet50 on my own. The classes are very different from each other so training needed and data quality needed is low.

# v1
The automobile_classifier_v1 will focus on KIA models seltos and carnival only
- The goal is to predict the KIA models with 90%+ accuracy. 
- The main task?
    - The dataset has to be very clean, with no noise, and no faulty data.
    - The classification classes are going to be very close as KIA follows similar designs among cars.
- Unfroze the layer4 with the model.fc

First run: 
![Confusion Matrix](utils\image.png)
- Carnival dataset had many U.S.A. variants of the carnival which looks very different from the Indian
- image size was set 

Second run:
![Confusion Matrix](utils\image-1.png)
- Seltos dataset had noise -> CLEARED

Third run: 
![Confusion Matrix](utils\image-2.png)
- Image size was changed to 384, and batch size to 32
- 20 epochs trained

Final Run:
![Confusion Matrix](utils\image-3.png)
- Increased the image size to 500, batch size to 64
- 20 epochs
- Dataset was highly cleaned - 400 to 500 images per class

# v2
- Added Kia sonet
- Added grad cam

# Models:
The models for each version are given in this drive link: [Automobile Classifier Models](https://drive.google.com/drive/folders/1Dzo2N9qPnRtoXx5FNe19c2tCIPrvAa0Y?usp=sharing)

# v3
- Added Kia EV6 data
- Increaed dataset from ~400 images to ~900 iamges for each class 
- total ~3600 images combined
- Increased number of epochs from 25 to 40
- Decreased learning rate from 1e-5 to 4e-6

![Confusion Matrix](utils\image-4.png)
![Loss and Accuracy Curve while training](utils\image-5.png)
- Time taken to train: 73mins - but I forced stop at 39th epoch as it started mild overfitting
``` 
--------------------------------------------------
Epoch [38/40]
Train Loss: 0.0358 | Train Accuracy: 99.27%
Val Loss:   0.0997 | Val Accuracy:   96.94%
--------------------------------------------------
```

# v4

***No model was trained, this version uses the final_model_v3 from version 3***

- Restructured repository and code
- Added checkpoints while training to notebook.py
- Made streamlit use full width instead of "centered"
- Added Inference Timing in streamlit page
- Added Prediction logs to track record each prediction made
- Added top-k prediction - Instead of prediction, it gives 1 top prediction, and 2nd and 3rd predictions according to confidence
- Model Statistics page added as a button to app.py

# v5
This version is specifically designed to test out different models and compare them with metrics.
A README file is given in the folder automobile_classifier_v5 explaining the structure of the code. 



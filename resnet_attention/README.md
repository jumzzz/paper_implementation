# Implementation of Residual Attention Network for Image Classification Paper

## Residual Attention Network Architecture
![Resnet Attention Network Architecture](images/resnet_attn_architecture.png)

## Implementation Details

#### Python Libraries
```
tensorflow               2.6.2
numpy                    1.19.5
matplotlib               3.5.0
python-magic             0.4.24
```

### Other Dependencies
- Kaggle API for acquiring the dataset (`pip install kaggle`)
- For reference of the API click [here](https://github.com/Kaggle/kaggle-api)

### Hardware Used

**GPU** : NVIDIA RTX 3060

#### Dataset Used
- [Weather Image Recognition](https://www.kaggle.com/jehanbhathena/weather-dataset) (Kaggle Dataset by Jehan Bhathena)
- **Image Classification Task with 11 Classes**

#### References
- [Residual Attention Network for Image Classification](https://arxiv.org/abs/1704.06904) (Original Paper by Fei Wang et.al.)
- [MixUp augmentation for image classification](https://keras.io/examples/vision/mixup/) (Keras Code Examples article by Sayak Paul)


#### Training Parameters:
- **Number of Training Epoch :** 29
- **Number of Batches:** 32

#### Implementation Result

- **Best Test Loss :** 0.9973
- **Best Test Accuracy :** 67.71% (11 Classes)


## Results of Training

### Train/Validation Loss for 11 Classes of Weather Image
![Training Loss](images/train_loss_v3.png)

### Train/Validation Accuracy for 11 Classes Weather Image
![Training Loss](images/train_accuracy_v3.png)
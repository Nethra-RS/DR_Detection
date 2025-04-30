# Diabetic Retinopathy Detection

This project uses **ResNet-50**, a deep convolutional neural network pretrained on ImageNet, and fine-tunes it using Keras/TensorFlow to classify retinal fundus images for Diabetic Retinopathy (DR) severity. The model categorizes images into 5 classes, from No DR to Proliferative DR, aiding early diagnosis and treatment.


## Contents

- Data loading and preprocessing
- Image augmentation using `ImageDataGenerator`
- Resnet-50 model design and training
- Visualization of training performance
- Model evaluation and predictions

## Requirements

To run this project, you’ll need Python 3.x and the following Python libraries:

```bash
pip install tensorflow keras numpy pandas matplotlib opencv-python scikit-learn seaborn


## Classes

| Label | Description         |
|-------|---------------------|
| 0     | No DR               |
| 1     | Mild DR             |
| 2     | Moderate DR         |
| 3     | Severe DR           |
| 4     | Proliferative DR    |


## Architecture

- Base Model:
  - `ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))`
  - Top (classification) layer is excluded to allow customization.
  - Partial freezing: The first 140 layers are frozen to preserve low-level image features.
  
- Custom Head:
  - `GlobalAveragePooling2D()` — replaces flattening for spatially invariant feature summarization.
  - `Dense(512, activation='relu')` — learns high-level representations.
  - `Dropout(0.4)` — reduces overfitting.
  - `Dense(256, activation='relu')`
  - `Dropout(0.3)`
  - Output layer - `Dense(5, activation='softmax')` — outputs class probabilities for the 5 DR stages.

- Compilation:
  - Optimizer: `Adam(learning_rate=0.0001)`
  - Loss: `categorical_crossentropy`
  - Metrics: `accuracy`


## Dataset

- Source: [Kaggle Diabetic Retinopathy 224x224 Gaussian Filtered](Dataset - https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered)
- Format: Images of retinal fundus with labels in a CSV file or class-based directories.
- Preprocessing:
  - Resize images to 224x224
  - Normalize pixel values - rescale to [0, 1]
  - Data augmentation to improve generalization (rotation, zoom, brightness shifts, shear)


### Train-Test Split

- The dataset is split into:
  - Training set: 80%
  - Validation set: 15%
  - Testing set: 5%


Note: Dataset is not included in this repository due to size. Please download it separately and place it in a `/data` directory.


## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Loss (categorical cross-entropy)
- Validation accuracy/loss plots

## Results

Sample results from training:

- Test Accuracy: 81.81%
- Validation Accuracy: 75%


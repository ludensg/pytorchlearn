# CIFAR-10 Classification using PyTorch

This project is a CIFAR-10 image classification experiment using PyTorch. The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. The goal is to train a convolutional neural network (CNN) to classify images into these classes.

## Structure

- **Model Architecture:** A custom CNN model is defined with convolutional and fully connected layers.
- **Data Loading:** The CIFAR-10 dataset is loaded and preprocessed.
- **Training:** The model is trained using the training dataset.
- **Evaluation:** The model's performance is evaluated on the test dataset.
- **Visualization:** Training and validation loss and accuracy are visualized.

## Files

- `cifar_experiment.py`: Contains the main code for model definition, training, and evaluation.

## Usage

1. **Environment Setup:**
   - Ensure that you have PyTorch and torchvision installed in your environment.
   - Additional libraries such as matplotlib are also required.

2. **Running the Script:**
   - Navigate to the directory containing the script.
   - Run the script using the command: `python cifar_experiment.py`

3. **Output:**
   - The script will output the training loss and accuracy.
   - A plot visualizing the training and validation loss and accuracy will be saved as `training_plots.png`.

## Model Architecture

The model consists of the following layers:

- Convolutional layers with ReLU activation functions.
- Max-pooling layers.
- Fully connected layers.

## Training and Evaluation

- **Optimizer:** Adam
- **Loss Function:** Cross-Entropy Loss
- **Metrics:** Accuracy

Training involves multiple epochs where the model learns from the training dataset. The model's performance is then evaluated based on the test dataset, and accuracy is used as the evaluation metric.
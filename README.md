
# Rice Image Classification using CNN

This repository contains the implementation of a Convolutional Neural Network (CNN) for classifying rice images from the [Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset). The model is designed using TensorFlow and Keras and achieves an accuracy of 90.16% on the test set.

## Model Architecture
The CNN model consists of a simple architecture with:
- 4 Convolutional Layer
- 4 MaxPooling Layer
- One Fully Connected Dense Layer

### Model Summary:
- **Convolutional Layer**: 32 filters, kernel size of 3x3, ReLU activation
- **Pooling Layer**: MaxPooling with a pool size of 2x2
- **Fully Connected Layer**: Dense layer with softmax activation for classification

## Training Details
- **Epochs**: 4
- **Batch Size**: As determined by the data generator
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

## Results
- **Test Accuracy**: 98.16%

## Dataset
The dataset used is the Rice Image Dataset available on Kaggle. It consists of several classes of rice images used for training, validation, and testing.


## Visualizations
The following plots show the training and validation accuracy and loss over the epochs:

### Model Accuracy
![Accuracy Plot](path/to/accuracy_plot.png)

### Model Loss
![Loss Plot](path/to/loss_plot.png)

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


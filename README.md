# Rice Image Classification Using CNN

Introduction¶
Rice is one of the most widely produced grains globally, with numerous genetic varieties. These varieties differ based on characteristics such as texture, shape, and color. By analyzing these distinguishing features, it is possible to classify rice varieties and assess seed quality.

In this study, five rice varieties commonly grown in Turkey—Arborio, Basmati, Ipsala, Jasmine, and Karacadag—were examined. The dataset includes 75,000 grain images, with 15,000 images from each variety.

To classify the data, Artificial Neural Network (ANN) and Deep Neural Network (DNN) algorithms were applied to the feature dataset, while the Convolutional Neural Network (CNN) algorithm was used for the image dataset. 

---

## Key Features  

### Dataset Handling  
- Efficiently loads images from subdirectories while handling hidden files and varying file formats.  

### Data Preprocessing  
- Resize all images to uniform dimensions.
- Moreover, we should resize the model 227*227 for AlexNet with 3 channels
- Normalizes pixel intensities for better model performance.  
- Splits the dataset into training, validation, and testing subsets.
- The Chi-Square test is used for analyzing the class distribution in image datasets, particularly when you want to check if the classes (labels) are evenly distributed. In image classification tasks, an imbalance in class distribution can lead to biased model performance. By applying the Chi-Square test to the label counts, you can detect potential imbalances and decide if data preprocessing like resampling or class weighting is necessary before training a model.

### Visualization  
- Displays dataset distributions and sample images for each rice type.  
- Generates statistics, such as image count, average pixel intensity, and aspect ratios.  

---

## Model Architecture  
The CNN model is designed with the following key components:  
- **Convolutional Layer**: 32 filters, kernel size 3x3, ReLU activation.  
- **Pooling Layer**: MaxPooling with a 2x2 pool size.  
- **Dropout**: To mitigate overfitting.  
- **Fully Connected Layer**: Dense layer with softmax activation for multi-class classification.  

### Training Details  
- **Epochs**: 10 
- **Batch Size**: Determined dynamically by the data generator.  
- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy  

### Training and Evaluation  
- Implements real-time data augmentation for robust learning.  
- Evaluates model accuracy and loss on the test dataset.  
- Visualizes training performance using accuracy and loss curves.  

---

## Results  
- **Test Accuracy**: **85.45%**  

---
## Evaluation of Multi-Class Image Classification Performance Using ROC Curve and AUC
A multi-class classification approach was applied using a synthetic dataset consisting of five classes, 1,000 samples, and 20 features. The dataset was divided into training and testing subsets, with 70% allocated for training and 30% for testing. A RandomForestClassifier model was trained on the training data and its performance was evaluated on the test data. To assess model performance, the ROC curve and AUC score were computed for each class. These metrics were derived by binarizing both the true and predicted labels, with the results visualized through ROC curve plots. The AUC values for each class were presented, offering a comprehensive evaluation of the classifier's performance across all categories.

## Dataset  
The dataset used is the **Rice Image Dataset**, available on Kaggle. It contains multiple classes of rice images, divided into training, validation, and testing subsets for classification tasks.  

---


## License  
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.  

--- 

This revision improves clarity, flow, and professionalism while ensuring all original details are preserved. Let me know if you'd like further adjustments!


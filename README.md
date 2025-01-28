# Rice Image Classification Using CNN  

This repository provides the implementation of a Convolutional Neural Network (CNN) for classifying rice images from the [Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset). The model, developed using TensorFlow and Keras, achieves a test accuracy of **98.61%**.  

---

## Key Features  

### Dataset Handling  
- Efficiently loads images from subdirectories while handling hidden files and varying file formats.  

### Data Preprocessing  
- Resizes all images to uniform dimensions.  
- Normalizes pixel intensities for better model performance.  
- Splits the dataset into training, validation, and testing subsets.  

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
- **Epochs**: 4  
- **Batch Size**: Determined dynamically by the data generator.  
- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy  

### Training and Evaluation  
- Implements real-time data augmentation for robust learning.  
- Evaluates model accuracy and loss on the test dataset.  
- Visualizes training performance using accuracy and loss curves.  

---

## Results  
- **Test Accuracy**: **98.61%**  

---

## Dataset  
The dataset used is the **Rice Image Dataset**, available on Kaggle. It contains multiple classes of rice images, divided into training, validation, and testing subsets for classification tasks.  

---


## License  
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.  

--- 

This revision improves clarity, flow, and professionalism while ensuring all original details are preserved. Let me know if you'd like further adjustments!


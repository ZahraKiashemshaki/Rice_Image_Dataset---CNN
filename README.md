# Rice Image Classification Using CNN

Introduction¶
Rice is one of the most widely produced grains globally, with numerous genetic varieties. These varieties differ based on texture, shape, and color. Analyzing these distinguishing features makes it possible to classify rice varieties and assess seed quality.

In this study, five rice varieties commonly grown in Turkey—Arborio, Basmati, Ipsala, Jasmine, and Karacadag—were examined. The dataset includes 75,000 grain images, with 15,000 images from each variety.

To classify the data, Artificial Neural Network (ANN) and Deep Neural Network (DNN) algorithms were applied to the feature dataset, while the Convolutional Neural Network (CNN) algorithm was used for the image dataset. 

---

## Key Features  

Certainly! Below is a professional and concise **README.md** template for your GitHub repository, summarizing everything you did with the dataset. You can customize it further based on your specific project details.

---

# Rice Image Classification

This repository contains the implementation of a **multi-class image classification model** to classify different types of rice using a dataset of rice images. The project includes data preprocessing, model training, evaluation, and prediction.

---

## **Dataset**
The dataset used in this project is the **Rice Image Dataset**, which contains images of five different types of rice:
- Basmati
- Ipsala
- Jasmine
- Arborio
- Karacadag

Each class has a set of images that were used for training and testing the model.

---

## **Project Overview**
The goal of this project is to build a deep learning model that can accurately classify rice images into their respective categories. The following steps were performed:

1. **Data Preprocessing**:
   - Images were resized to `(64, 64)` pixels.
   - Data augmentation techniques (e.g., rotation, flipping, zooming) were applied to increase dataset diversity and reduce overfitting.
   - The dataset was split into training (80%) and validation (20%) sets.

2. **Model Architecture**:
   - A **Convolutional Neural Network (CNN)** was implemented using TensorFlow/Keras.
   - The model consists of:
     - Two convolutional blocks with `MaxPooling2D` layers.
     - A `GlobalAveragePooling2D` layer to reduce spatial dimensions.
     - A fully connected layer with `Dropout` for regularization.
     - A `softmax` output layer for multi-class classification.

3. **Model Training**:
   - The model was compiled using the **Adam optimizer** and **categorical cross-entropy loss**.
   - Training was performed for 10 epochs with **early stopping** to prevent overfitting.
   - Data augmentation was applied during training to improve generalization.

4. **Model Evaluation**:
   - The model achieved a **test accuracy of 94.5%**.
   - ROC curves and AUC scores were computed to evaluate the model's performance for each class.

5. **Prediction**:
   - A function was implemented to predict the class of a given rice image.
   - The model successfully predicted the class of unseen images.

---

## **Code Structure**
- **`data_preprocessing.py`**: Script for loading and preprocessing the dataset.
- **`model_training.py`**: Script for defining, compiling, and training the model.
- **`evaluation.py`**: Script for evaluating the model using ROC curves and AUC.
- **`predict.py`**: Script for predicting the class of a new rice image.

---

## **Results**
- **Test Accuracy**: 94.5%
- **ROC Curves**: Visualized for each class to demonstrate the model's ability to distinguish between rice types.
- **Prediction Example**:
  ```
  Trying to load image from: /content/rice_data/Rice_Image_Dataset/Rice_Image_Dataset/Basmati/basmati (11218).jpg
  Predicted Class: Basmati
  ```

---

## **Requirements**
To run the code, you need the following Python libraries:
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

Install the dependencies using:
```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

---

## **Usage**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rice-image-classification.git
   cd rice-image-classification
   ```

2. Run the scripts in the following order:
   - Data preprocessing: `python data_preprocessing.py`
   - Model training: `python model_training.py`
   - Evaluation: `python evaluation.py`
   - Prediction: `python predict.py`

---

## **Future Work**
- Experiment with more advanced architectures (e.g., ResNet, EfficientNet).
- Fine-tune hyperparameters for better performance.
- Deploy the model as a web application using Flask or Streamlit.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to modify this template to better suit your project! Let me know if you need further assistance.

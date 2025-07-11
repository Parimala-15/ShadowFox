# 🧠 Image Tagging with CNN – CIFAR-10 Classification

This project builds an image classification model using a **Convolutional Neural Network (CNN)** to categorize images from the **CIFAR-10 dataset** into 10 basic classes such as **cat**, **dog**, **car**, etc. The solution is implemented using **PyTorch** and demonstrates best practices in training, evaluation, and deployment readiness.

---

## 📌 Problem Statement

Design a practical image tagging solution using machine learning that can classify images into simple, everyday object categories like cat, dog, car, ship, etc. The model should be efficient, generalizable, and suitable for real-world applications.

---

## 📂 Dataset

- **Source**: CIFAR-10 dataset from `torchvision.datasets`
- **Classes**: 
['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

- **Images**: 60,000 color images (32x32 pixels)
- 50,000 for training (split into train and validation)
- 10,000 for testing

---

## 🔍 Workflow Summary

1. **Device Detection**
 - Automatically detects CUDA / MPS / CPU for training
 - Ensures compatibility across platforms

2. **Data Preprocessing**
 - Normalized pixel values
 - Applied transformations to convert PIL images to tensors

3. **Image Augmentation**
 - Used during training to improve generalization:
   - Random horizontal flip
   - Random cropping
   - Color jittering

4. **Train-Validation Split**
 - 80% for training, 20% for validation (from original train set)

5. **CNN Model Architecture**
 - 3 Convolutional layers with ReLU, MaxPooling
 - Fully connected layers
 - LogSoftmax output activation
 - Dropout and BatchNorm for regularization

6. **Training Loop**
 - Optimizer: Adam
 - Loss Function: NLLLoss
 - Tracks training loss and validation loss for every epoch

7. **Evaluation**
 - Accuracy on test set
 - Classification Report (Precision, Recall, F1-Score)
 - Confusion Matrix for class-wise performance

8. **Model Saving**
 - Trained model is saved as `cnn_cifar10.pth`

---

## 📈 Model Performance

- ✅ **Achieved Accuracy**: ~84% on test set
- 📊 Detailed metrics via `classification_report` and `confusion_matrix`
- 🧠 Robust performance with low overfitting due to validation monitoring and data augmentation

---

## 🛠 Tools & Libraries

- Python
- PyTorch & Torchvision
- Matplotlib, Seaborn, NumPy
- Scikit-learn



## 📁 Files in This Repo

| File | Description |
|------|-------------|
| `cnn_cifar10.pth` | Trained model weights |
| `model.py` | CNN model definition |
| `train.py` | Model training and validation code |
| `evaluate.py` | Evaluation metrics and visualizations |
| `utils.py` | Utility functions for prediction |
| `README.md` | Project overview and documentation |




## 🧠 Future Improvements
Add EarlyStopping and Learning Rate Schedulers
Try deeper models (ResNet18, EfficientNet)
Experiment with custom datasets or real-time webcam inputs
Deploy as a web app using Flask or Streamlit


# CNN Image Classifier — CIFAR-10

## 📌 Project Overview
This project builds and trains a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into 10 categories. It demonstrates image preprocessing, model construction, training loops, evaluation, and visualization.

## 🔍 Problem Statement
The goal of this project is to correctly classify small color images into categories such as airplanes, birds, cats, dogs, cars, and more. The challenge is training a deep model that generalizes well on a diverse dataset.

## 🧠 Approach & Methodology
- Loaded CIFAR-10 dataset through PyTorch  
- Applied normalization and augmentation  
- Built a CNN with convolution, ReLU, pooling, and fully connected layers  
- Trained with Adam optimizer  
- Evaluated using accuracy and training loss  

## 🧪 Results & Evaluation
Placeholder metrics (since training varies):
- Training accuracy reached: **~75–80%**
- Loss decreased consistently over epochs  

## 📚 Learning Outcomes
During this project, I learned:  
- How CNNs process visual data  
- How to build a full deep learning pipeline  
- How to evaluate image model performance  

## ▶️ How to Run the Code
```bash
pip install -r requirements.txt
python train.py

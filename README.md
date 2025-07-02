# AI-model-training
# 📊 Machine Learning & NLP Projects

This repository contains three mini-projects that demonstrate the application of modern AI tools using TensorFlow, Scikit-learn, and spaCy. Each project is well-commented and organized for learning and experimentation.

---

## 🔢 1. Handwritten Digit Recognition with CNN (TensorFlow)

**Goal**: Train a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset.

### Features
- Uses TensorFlow and Keras
- Data augmentation with `ImageDataGenerator`
- Model evaluation using accuracy, classification report, and confusion matrix
- Model saved for future use

### Steps
1. Load and normalize MNIST dataset
2. Apply data augmentation
3. Build and train a CNN
4. Evaluate on test data and visualize confusion matrix
5. Save the model (`my_digit_model.keras`)

---

## 🌼 2. Iris Species Classification (Scikit-learn)

**Goal**: Train a Decision Tree classifier to identify species of Iris flowers.

### Features
- Uses `sklearn`’s built-in Iris dataset
- Handles data preprocessing and splitting
- Trains a Decision Tree Classifier
- Evaluates performance with accuracy, precision, and recall

### Steps
1. Load Iris dataset
2. Preprocess and split into training/testing sets
3. Train the classifier
4. Evaluate using Scikit-learn metrics

---

## 🗣️ 3. Named Entity Recognition & Sentiment Analysis (spaCy)

**Goal**: Perform Named Entity Recognition (NER) and rule-based sentiment analysis on Amazon product reviews.

### Features
- Uses `spaCy` for NER to extract product names and brands
- Basic rule-based sentiment detection (positive/negative)
- Parses and analyzes multiple reviews

### Steps
1. Load the `en_core_web_sm` spaCy model
2. Extract entities related to products/brands from reviews
3. Use rule-based logic to assign sentiment labels
4. Output named entities and sentiment result

---

## 📁 Project Structure


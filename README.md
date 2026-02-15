# PBL-Project
# 🔍 Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-98.53%25-brightgreen.svg)

A machine learning-based system that automatically classifies news articles as **real** or **fake** using Natural Language Processing (NLP) and Logistic Regression. Achieves **98.53% accuracy** on test data.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## 🌟 Overview

The rapid spread of misinformation on digital platforms poses a significant threat to public trust and informed decision-making. This project implements an automated fake news detection system using machine learning techniques to identify fabricated news articles with high accuracy.

### Key Highlights
- ✅ **98.53%** accuracy on test dataset
- ✅ **99%** precision for fake news detection
- ✅ **99%** recall for real news classification
- ✅ Lightweight and computationally efficient
- ✅ Real-time classification capability

---

## 🎯 Problem Statement

Manual fact-checking cannot scale with the massive volume of content generated daily on social media and news platforms. This project addresses the critical need for an **automated, accurate, and efficient system** to identify fake news articles before they can cause widespread harm.

### Research Gap
Existing solutions often rely on:
- **Rule-based systems** (limited flexibility)
- **Deep learning models** (high computational cost)
- **Manual fact-checking** (not scalable)

Our approach combines **TF-IDF vectorization** with **Logistic Regression** to achieve optimal balance between accuracy and computational efficiency.

---

## ✨ Features

- **Text Preprocessing**: Automated cleaning and normalization of news articles
- **Feature Extraction**: TF-IDF vectorization to capture semantic patterns
- **Binary Classification**: Distinguishes between fake (0) and real (1) news
- **High Performance**: 98.53% accuracy with minimal false positives
- **Scalable**: Processes articles in real-time without GPU requirements
- **Interpretable**: Logistic regression provides explainable predictions

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.8+ |
| **ML Framework** | scikit-learn |
| **Data Processing** | pandas, NumPy |
| **Text Processing** | regex, TfidfVectorizer |
| **Evaluation** | scikit-learn metrics |

### Core Libraries
```python
pandas==1.5.0
numpy==1.23.0
scikit-learn==1.1.0
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
Place `Fake.csv` and `True.csv` files in the `dataset/` directory.

---

## 💻 Usage

### Training the Model

```python
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

# Label data
fake["label"] = 0
real["label"] = 1

# Combine and shuffle
df = pd.concat([fake, real])
df = df.sample(frac=1).reset_index(drop=True)

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

df["text"] = df["text"].apply(clean_text)

# Split data
X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

### Making Predictions

```python
# Sample prediction
sample_text = "Breaking news: Major political scandal exposed!"
cleaned = clean_text(sample_text)
vectorized = vectorizer.transform([cleaned])
prediction = model.predict(vectorized)

if prediction[0] == 0:
    print("⚠️ FAKE NEWS DETECTED")
else:
    print("✓ REAL NEWS")
```

---

## 📊 Dataset

### Data Sources
- **Fake.csv**: Collection of fabricated news articles
- **True.csv**: Collection of authentic news articles

### Data Preprocessing
1. **Lowercasing**: Convert all text to lowercase
2. **Special Character Removal**: Remove non-alphanumeric characters
3. **Tokenization**: Split text into individual words
4. **Stop Words Removal**: Remove common English words
5. **TF-IDF Vectorization**: Convert text to numerical features

### Data Split
- **Training Set**: 75% (33,675 samples)
- **Test Set**: 25% (11,225 samples)

### Class Distribution
| Class | Label | Count |
|-------|-------|-------|
| Fake News | 0 | 5,870 (test) |
| Real News | 1 | 5,355 (test) |

---

## 🏗️ Model Architecture

### Pipeline
```
Raw Text → Preprocessing → TF-IDF Vectorization → Logistic Regression → Prediction
```

### Components

#### 1. **Text Preprocessing**
- Converts text to lowercase
- Removes special characters and punctuation
- Normalizes whitespace

#### 2. **TF-IDF Vectorizer**
- **Parameters**:
  - `stop_words="english"`: Removes common English words
  - `max_df=0.7`: Ignores terms appearing in >70% of documents
- **Output**: Sparse matrix of TF-IDF features

#### 3. **Logistic Regression Classifier**
- **Algorithm**: Binary logistic regression
- **Optimization**: LBFGS solver (default)
- **Regularization**: L2 penalty
- **Output**: Binary classification (0=Fake, 1=Real)

---

## 📈 Results

### Performance Metrics

| Metric | Fake News (0) | Real News (1) | Overall |
|--------|---------------|---------------|---------|
| **Precision** | 99% | 98% | 98.5% |
| **Recall** | 99% | 99% | 99% |
| **F1-Score** | 99% | 98% | 98.5% |
| **Accuracy** | - | - | **98.53%** |

### Classification Report
```
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      5870
           1       0.98      0.99      0.98      5355

    accuracy                           0.99     11225
   macro avg       0.99      0.99      0.99     11225
weighted avg       0.99      0.99      0.99     11225
```

### Key Achievements
- ✅ Minimal false positives (1% error rate for fake news)
- ✅ High recall ensures real news is rarely misclassified
- ✅ Balanced performance across both classes
- ✅ Computationally efficient (no GPU required)

### Comparison with Baselines

| Approach | Accuracy |
|----------|----------|
| Rule-based systems | ~70-75% |
| Traditional ML (Naive Bayes) | ~85-90% |
| **Our Solution (LR + TF-IDF)** | **98.53%** |
| Deep Learning (LSTM) | ~96-98% (higher cost) |

---

## 📁 Project Structure

```
fake-news-detection/
│
├── dataset/
│   ├── Fake.csv              # Fake news dataset
│   └── True.csv              # Real news dataset
│
├── notebooks/
│   └── dataset2.ipynb        # Main training notebook
│
├── models/
│   ├── model.pkl             # Trained Logistic Regression model
│   └── vectorizer.pkl        # Fitted TF-IDF vectorizer
│
├── src/
│   ├── preprocess.py         # Text preprocessing utilities
│   ├── train.py              # Model training script
│   └── predict.py            # Prediction script
│
├── presentation/
│   └── pbl_presentation_fake_news.html  # Project presentation
│
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── LICENSE                   # MIT License
```

---

## 🔮 Future Enhancements

### Short-term
- [ ] Add web scraping for real-time news article collection
- [ ] Implement model persistence (save/load trained models)
- [ ] Create REST API for predictions
- [ ] Build web interface for user interaction

### Medium-term
- [ ] Multi-class classification (fake, real, satire, opinion)
- [ ] Implement additional features (source credibility, author history)
- [ ] Add explainability (LIME/SHAP for feature importance)
- [ ] Deploy as cloud service (AWS/GCP/Azure)

### Long-term
- [ ] Experiment with transformer models (BERT, RoBERTa)
- [ ] Multi-lingual support
- [ ] Real-time monitoring dashboard
- [ ] Mobile application development
- [ ] Integration with browser extensions

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Contribution Guidelines
- Follow PEP 8 style guide for Python code
- Add tests for new features
- Update documentation accordingly
- Ensure all tests pass before submitting PR

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Project Guide**: Mr. Lav Upadhyay
- **Developer**: Kanisha Agrawal (2427030280)
- **Institution**: Manipal University Jaipur, Department of Computer Science & Engineering
- **Dataset**: Kaggle Fake News Dataset
- **Libraries**: scikit-learn, pandas, NumPy

---

## 📞 Contact

**Kanisha Agrawal**
- Registration No: 2427030280
- Institution: Manipal University Jaipur
- Project Guide: Mr. Lav Upadhyay

---

## 📊 Statistics

![GitHub stars](https://img.shields.io/github/stars/yourusername/fake-news-detection?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/fake-news-detection?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/fake-news-detection)

---

## 🔗 References

1. scikit-learn Documentation: https://scikit-learn.org/
2. TF-IDF Vectorization: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
3. Logistic Regression: https://en.wikipedia.org/wiki/Logistic_regression
4. Fake News Detection Research Papers (add relevant papers)

---

<div align="center">

**Made with ❤️ by Kanisha Agrawal**

⭐ Star this repository if you find it helpful!

</div>

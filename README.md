# OBF-Psychiatric ML Classifiers
This repository provides an optimized machine learning pipeline for classifying psychiatric conditions using the OBF-Psychiatric dataset (motor activity data from patients with depression, schizophrenia, ADHD, and controls).

## 🔹 Features
- **Feature engineering** techniques to extract temporal patterns
- Multiple **classifiers** (Random Forest, SVM, XGBoost, Logistic Regression, KNN, VotingClassifier....)  
- **GPU acceleration** with cuML & XGBoost  
- **Optimized hyperparameter tuning** (RandomizedSearchCV)  
- **Cross-validation & performance evaluation** (confusion matrices, classification reports)  
- Multiple **Deep Learning models** (LSTMModel, CNN1DModel, TransformerModel...)
## 🚀 Installation & Usage
1. Clone the repo:  
```bash
git clone https://github.com/atinak/obf-psychiatric.git
cd obf-psychiatric
Install dependencies:
```
```bash

conda create -n obf-ml python=3.10
conda activate obf-ml
pip install -r requirements.txt



## 📌 About the dataset:
The OBF-Psychiatric dataset consists of actigraphy-based motor activity recordings from patients diagnosed with psychiatric disorders, making it useful for mental health research & digital biomarkers.
The original data can be found in:
Garcia-Ceja, E., Stautland, A., Riegler, M.A. et al. OBF-Psychiatric, a motor activity dataset of patients diagnosed with major depression, schizophrenia, and ADHD. Sci Data 12, 32 (2025). https://doi.org/10.1038/s41597-025-04384-3



## 🔬 Future Improvements
Fix some minor bugs 

Optimize inference speed for real-time applications

Enhance dataset preprocessing & augmentation strategies

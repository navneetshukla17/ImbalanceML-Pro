# ImbalanceML Pro

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live-Demo-success.svg)](https://navneet-shukla-imbalance-ml-pro.streamlit.app/)

**A Professional Machine Learning Tool for Imbalanced Classification Problems**


## 🌐 Live Demo

**Experience ImbalanceML Pro instantly without any setup:**

### 👉 [Launch Live Application](https://navneet-shukla-imbalance-ml-pro.streamlit.app/)

No installation, no configuration - just click and start building models!

---

## 📋 Project Overview

**ImbalanceML Pro** is a comprehensive, production-ready machine learning platform specifically designed to handle imbalanced classification problems. Built with Streamlit, it provides an intuitive interface for data scientists and ML practitioners to build, evaluate, and deploy models that excel at detecting rare events in highly imbalanced datasets.

### 🎯 Key Features

- **Multiple ML Algorithms**: Support for Logistic Regression, Decision Trees, Random Forest, SVM, Bagging Ensembles, AdaBoost, Gradient Boosting, and XGBoost
- **Advanced Preprocessing**: Configurable feature scaling, encoding, and missing value handling strategies
- **Imbalance Handling**: Built-in support for SMOTE, Random Oversampling, Undersampling, and Class Weights
- **Comprehensive Evaluation**: Multiple metrics including Precision, Recall, F1-Score, ROC-AUC, and PR-AUC
- **Interactive Visualizations**: Confusion matrices, ROC curves, Precision-Recall curves, and feature importance plots
- **Model Explainability**: SHAP and LIME integration for interpretable AI
- **Experiment Tracking**: Full experiment logging and comparison dashboard
- **Built-in Datasets**: 5 pre-loaded synthetic datasets for quick prototyping
- **Model Export**: Save and download trained models with metadata

---

## 🚀 Quick Start

### Option 1: Use Live Demo (Recommended)
Simply visit the [live application](https://navneet-shukla-imbalance-ml-pro.streamlit.app/) and start working immediately.

### Option 2: Local Installation

#### Prerequisites

- Python 3.8 or higher
- pip package manager

#### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/imbalanceml-pro.git
cd imbalanceml-pro
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

#### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

---

## 📁 Project Structure

```
imbalanceml-pro/
│
├── app.py                    # Main Streamlit application
├── preprocessing.py          # Data preprocessing utilities
├── training.py              # Model training and evaluation
├── visualizations.py        # Visualization engine
├── datasets.py              # Built-in dataset manager
├── experiment_tracker.py    # Experiment logging system
├── utils.py                 # Utility functions
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
└── LICENSE                 # License file
```

---

## 🔧 Core Modules

### 1. **app.py** - Main Application
The central hub of the application providing:
- Interactive UI with 6 main tabs
- Session state management
- Navigation and workflow control
- Integration of all components

### 2. **preprocessing.py** - Data Preprocessing
Handles all data preparation tasks:
- **Scaling Methods**: Standard, MinMax, Robust, Normalize
- **Encoding**: Label, One-Hot, Ordinal
- **Missing Values**: Mean, Median, Most Frequent, KNN imputation
- **Imbalance Handling**: SMOTE, Random Over/Undersampling

### 3. **training.py** - Model Training
Manages model training and evaluation:
- Multiple algorithm implementations
- Hyperparameter tuning (Grid Search, Random Search)
- Stratified K-Fold cross-validation
- Comprehensive metrics calculation

### 4. **visualizations.py** - Visualization Engine
Creates interactive plots:
- Confusion matrices
- ROC and PR curves
- Feature importance charts
- SHAP/LIME explanations

### 5. **datasets.py** - Dataset Manager
Provides built-in datasets:
- Credit Card Fraud Detection
- Customer Churn Prediction
- Email Spam Classification
- Medical Diagnosis
- Synthetic Imbalanced Data

### 6. **experiment_tracker.py** - Experiment Tracking
Logs and compares experiments:
- UUID-based experiment identification
- Performance metrics storage
- Model comparison dashboard
- CSV/JSON export capabilities

---

## 📊 Usage Guide

### Step 1: Data Upload
- Upload your CSV/XLSX file, or
- Select from 5 built-in datasets
- Preview data and check class distribution

### Step 2: Configure Training
- Select model algorithm
- Choose preprocessing options:
  - Feature scaling method
  - Categorical encoding strategy
  - Missing value handling
- Configure imbalance handling
- Set hyperparameters

### Step 3: Train Model
- Click "Train Model"
- Monitor training progress
- View cross-validation results

### Step 4: Analyze Results
- Review performance metrics
- Explore visualization plots
- Compare with previous experiments

### Step 5: Explain Predictions
- Generate SHAP summaries
- Create LIME explanations
- Understand feature importance

### Step 6: Export Model
- Download trained model (.pkl)
- Include metadata and preprocessing steps

---

## 🎓 Built-in Datasets

### 1. Credit Card Fraud Detection
- **Samples**: 5,000
- **Features**: 10 (transaction amount, account age, merchant category, etc.)
- **Imbalance**: 97:3 (Normal:Fraud)

### 2. Customer Churn Prediction
- **Samples**: 3,000
- **Features**: 12 (tenure, charges, contract type, support calls, etc.)
- **Imbalance**: 75:25 (Stay:Churn)

### 3. Email Spam Detection
- **Samples**: 4,000
- **Features**: 12 (email length, links, sender reputation, etc.)
- **Imbalance**: 85:15 (Ham:Spam)

### 4. Medical Diagnosis
- **Samples**: 2,500
- **Features**: 12 (age, BMI, blood pressure, lifestyle factors, etc.)
- **Imbalance**: 80:20 (Healthy:Disease)

### 5. Synthetic Imbalanced
- **Samples**: 3,000
- **Features**: 20 (generated features)
- **Imbalance**: 90:10 (Majority:Minority)

---

## 🔬 Evaluation Metrics

ImbalanceML Pro provides comprehensive evaluation metrics suitable for imbalanced datasets:

- **Accuracy**: Overall correctness (use with caution on imbalanced data)
- **Precision**: Positive predictive value
- **Recall**: Sensitivity, true positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **PR-AUC**: Area under the Precision-Recall curve (preferred for imbalanced data)

---

## 🧠 Model Explainability

### SHAP (SHapley Additive exPlanations)
- Summary plots showing global feature importance
- Waterfall plots for individual predictions
- Force plots for detailed breakdowns

### LIME (Local Interpretable Model-agnostic Explanations)
- Local explanations for individual predictions
- Feature contribution analysis
- Model-agnostic approach

---

## 📈 Experiment Tracking

The experiment tracker provides:
- **Automatic Logging**: Every trained model is logged with UUID
- **Comparison Dashboard**: Compare multiple models side-by-side
- **Performance Ranking**: Composite scores across metrics
- **Visual Comparison**: Radar charts and bar plots
- **Export Options**: Download experiment logs as CSV

---

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **ML Libraries**: scikit-learn, XGBoost, imbalanced-learn
- **Visualization**: Plotly, Matplotlib
- **Explainability**: SHAP, LIME
- **Data Processing**: Pandas, NumPy

---

## 📦 Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
plotly>=5.17.0
xgboost>=2.0.0
shap>=0.43.0
lime>=0.2.0
joblib>=1.3.0
openpyxl>=3.1.0
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Navneet Shukla**

- LinkedIn: [@navneet-shukla17](https://www.linkedin.com/in/navneet-shukla17/)
- Project Live Demo: [ImbalanceML Pro](https://navneet-shukla-imbalance-ml-pro.streamlit.app/)

---

## 🙏 Acknowledgments

- scikit-learn documentation and tutorials
- imbalanced-learn for imbalance handling techniques
- Streamlit for the amazing framework
- SHAP and LIME authors for explainability tools

---

## 📞 Support

- **Issues**: Open an issue in the [GitHub repository](https://github.com/yourusername/imbalanceml-pro/issues)
- **Questions**: Connect on [LinkedIn](https://www.linkedin.com/in/navneet-shukla17/)
- **Live Demo**: [Try the application](https://navneet-shukla-imbalance-ml-pro.streamlit.app/)

---

## 🗺️ Roadmap

- [ ] Add deep learning models (Neural Networks)
- [ ] Support for multi-class imbalanced classification
- [ ] Advanced ensemble methods
- [ ] AutoML capabilities
- [ ] Cloud deployment guides
- [ ] REST API for model serving
- [ ] Real-time prediction interface

---

## ⭐ Show Your Support

If you find this project helpful, please consider giving it a star on GitHub!

---

<div align="center">

**Made with ❤️ for the ML Community**

[Live Demo](https://navneet-shukla-imbalance-ml-pro.streamlit.app/) • [Report Bug](https://github.com/yourusername/imbalanceml-pro/issues) • [Request Feature](https://github.com/yourusername/imbalanceml-pro/issues)

</div>

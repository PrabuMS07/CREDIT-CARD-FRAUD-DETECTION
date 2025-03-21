# Credit Card Fraud Detection

## Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. Two models are implemented: Logistic Regression and Random Forest Classifier. The dataset used for training is highly imbalanced, so the SMOTE (Synthetic Minority Over-sampling Technique) method is applied to balance the class distribution.

## Dataset
The dataset used in this project is available on Kaggle:
[Download Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Dataset Description:
- The dataset contains credit card transactions labeled as fraudulent (1) or non-fraudulent (0).
- Features include 28 anonymized principal components obtained via PCA, along with `Time` and `Amount`.

## Installation
Ensure you have Python installed, along with the required dependencies. You can install them using the following command:

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

## Project Structure
- `creditcard.csv` - The dataset file (download separately)
- `fraud_detection.py` - The main script for training and evaluating models

## Steps in the Code
1. **Load the Dataset**: Reads `creditcard.csv` into a Pandas DataFrame.
2. **Preprocess Data**:
   - Check for missing values.
   - Separate features (`X`) and target variable (`y`).
   - Standardize the features using `StandardScaler`.
   - Apply SMOTE to handle class imbalance.
3. **Split Data**: The resampled dataset is split into training (80%) and testing (20%) sets.
4. **Train Models**:
   - Logistic Regression
   - Random Forest Classifier
5. **Evaluate Models**:
   - Predictions are made on the test set.
   - Performance metrics such as Precision, Recall, F1 Score, and Classification Report are computed.

## Running the Code
Run the script using the following command:

```bash
python fraud_detection.py
```

## Model Performance Metrics
After execution, the script outputs precision, recall, and F1 scores for both models, along with classification reports.

## Future Enhancements
- Try additional models such as XGBoost, Neural Networks.
- Implement feature engineering for better model performance.
- Use anomaly detection techniques.

## License
This project is open-source and available for educational purposes.


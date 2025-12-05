# ONCFM: Fake Banknote Detection Project Report

## 1. Executive Summary
The Organisation Nationale de Lutte Contre le Faux-Monnayage (ONCFM) requires a robust system to detect counterfeit euro banknotes based on geometric dimensions. This project analyzed a dataset of 1500 banknotes, performed data cleaning and exploratory analysis, and evaluated multiple machine learning models.

**Conclusion:** A **Logistic Regression** model was selected as the final solution due to its high accuracy (99% on test set) and interpretability. A deployment script `predict_banknote.py` has been developed to allow agents to check banknotes in the field.

## 2. Data Description & Preparation
The dataset `billets.csv` contains 1500 entries with the following features:
- **diagonal**: Diagonal length of the banknote.
- **height_left**: Height measured at the left.
- **height_right**: Height measured at the right.
- **margin_low**: Lower margin width.
- **margin_up**: Upper margin width.
- **length**: Total length of the banknote.
- **is_genuine**: Target variable (True = Real, False = Fake).

### Data Cleaning
- **Missing Values**: The `margin_low` feature contained missing values.
- **Imputation Strategy**: A Linear Regression model was trained using the other features to predict and fill the missing `margin_low` values, ensuring data integrity without dropping valuable records.

### Preprocessing
- **Scaling**: All features were standardized using `StandardScaler` to ensure zero mean and unit variance, which is critical for distance-based algorithms like K-Means and KNN, and for the convergence of Logistic Regression.
- **Split**: Data was split into Training (80%) and Test (20%) sets, stratified by the target variable to maintain class balance.

## 3. Exploratory Data Analysis (EDA)
Key insights from the analysis:
- **Separability**: Visualizations (scatter plots) showed a clear separation between genuine and fake banknotes, particularly when combining `margin_low` and `length`.
- **Correlations**: Strong correlations were observed between margin dimensions and the authenticity of the bill.

## 4. Modeling & Comparison
Four algorithms were tested:
1.  **K-Means Clustering**: Used as an unsupervised baseline to see if data naturally clusters into two groups matching the labels.
2.  **Logistic Regression**: A linear classifier that provides probabilities.
3.  **K-Nearest Neighbors (KNN)**: A distance-based classifier.
4.  **Random Forest**: An ensemble method for capturing non-linear relationships.

### Performance
| Model | Accuracy | Interpretation |
|-------|----------|----------------|
| **Logistic Regression** | **99.0%** | **Selected.** High accuracy, fast, and provides interpretable coefficients. |
| Random Forest | ~99% | Excellent but more complex and harder to interpret. |
| KNN | ~98% | Good, but sensitive to outliers and slower at inference. |
| K-Means | N/A | Good separation observed, validating the distinctness of classes. |

**Why Logistic Regression?**
It offers the perfect balance of performance and simplicity. The coefficients allow us to understand exactly which features contribute to a "Fake" classification (e.g., larger margins often indicate a fake).

## 5. Deployment
A standalone script `predict_banknote.py` is provided for operational use.

### How to Run
**Option 1: Interactive Mode**
Run the script without arguments and follow the prompts:
```bash
python predict_banknote.py
```

**Option 2: Command Line Arguments**
Pass the dimensions directly:
```bash
python predict_banknote.py --diagonal 171.81 --height_left 104.86 --height_right 104.95 --margin_low 4.52 --margin_up 2.89 --length 112.83
```

### Output
The script returns:
- **Prediction**: "Real Banknote" or "Fake Banknote"
- **Probability**: The confidence score (e.g., "Probability of being Real: 73.92%")

## 6. Recommendations
- **Regular Retraining**: As new counterfeits appear, the model should be retrained with new data.
- **Field Testing**: The probability score should be used to flag "uncertain" cases (e.g., 40-60% probability) for manual expert inspection.

---

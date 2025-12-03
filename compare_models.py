import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, roc_auc_score, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING AND PREPARATION
# ============================================================================
print("Chargement et préparation des données...")
try:
    df = pd.read_csv('datasets/billets.csv', sep=';')
except FileNotFoundError:
    print("Erreur: Le fichier 'datasets/billets.csv' est introuvable.")
    exit()

# Imputation des valeurs manquantes (Régression Linéaire)
df_complete = df[df['margin_low'].notna()]
df_missing = df[df['margin_low'].isna()]

X_train_impute = df_complete[['diagonal', 'height_left', 'height_right', 'margin_up', 'length']]
y_train_impute = df_complete['margin_low']

model_impute = LinearRegression().fit(X_train_impute, y_train_impute)

df.loc[df['margin_low'].isna(), 'margin_low'] = model_impute.predict(
    df_missing[['diagonal', 'height_left', 'height_right', 'margin_up', 'length']]
)

# Préparation des features et target
X = df.drop('is_genuine', axis=1)
y = df['is_genuine'].astype(int) # 1 for True, 0 for False

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# ============================================================================
# 2. MODEL DEFINITION
# ============================================================================
print("Définition des modèles...")

# Custom scorer for Fake Bill Recall (Class 0)
# Note: In sklearn, recall_score with pos_label=0 calculates recall for the negative class
scoring = {
    'accuracy': 'accuracy',
    'recall_fake': make_scorer(recall_score, pos_label=0),
    'precision_fake': make_scorer(precision_score, pos_label=0),
    'f1_fake': make_scorer(f1_score, pos_label=0),
    'roc_auc': 'roc_auc'
}

models = {
    'Logistic Regression (Standard)': LogisticRegression(random_state=42),
    'Logistic Regression (Optimized)': LogisticRegression(random_state=42, class_weight={0: 3.0, 1: 1.0}), # Based on previous optimization
    'KNN (k=10)': KNeighborsClassifier(n_neighbors=10),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

# ============================================================================
# 3. CROSS-VALIDATION COMPARISON
# ============================================================================
print("Lancement de la validation croisée (10-fold)...")

results = []
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"  Evaluation de {name}...")
    cv_results = cross_validate(model, X_scaled, y, cv=cv, scoring=scoring)
    
    for i in range(10):
        results.append({
            'Model': name,
            'Fold': i,
            'Accuracy': cv_results['test_accuracy'][i],
            'Recall (Fake)': cv_results['test_recall_fake'][i],
            'Precision (Fake)': cv_results['test_precision_fake'][i],
            'F1-Score (Fake)': cv_results['test_f1_fake'][i],
            'ROC AUC': cv_results['test_roc_auc'][i]
        })

df_results = pd.DataFrame(results)

# ============================================================================
# 4. RESULTS AGGREGATION AND DISPLAY
# ============================================================================
print("\n" + "="*80)
print("RÉSULTATS MOYENS (10-FOLD CV)")
print("="*80)

summary = df_results.groupby('Model').agg({
    'Recall (Fake)': ['mean', 'std'],
    'Accuracy': ['mean', 'std'],
    'F1-Score (Fake)': ['mean', 'std'],
    'ROC AUC': ['mean', 'std']
}).sort_values(('Recall (Fake)', 'mean'), ascending=False)

print(summary)

# ============================================================================
# 5. VISUALIZATION
# ============================================================================
print("\nGénération des graphiques...")

plt.figure(figsize=(14, 8))

# Recall Boxplot
plt.subplot(2, 2, 1)
sns.boxplot(x='Model', y='Recall (Fake)', data=df_results)
plt.title('Distribution du Recall (Faux Billets) par Modèle')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Accuracy Boxplot
plt.subplot(2, 2, 2)
sns.boxplot(x='Model', y='Accuracy', data=df_results)
plt.title('Distribution de l\'Accuracy par Modèle')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# F1-Score Boxplot
plt.subplot(2, 2, 3)
sns.boxplot(x='Model', y='F1-Score (Fake)', data=df_results)
plt.title('Distribution du F1-Score (Faux Billets) par Modèle')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# ROC AUC Boxplot
plt.subplot(2, 2, 4)
sns.boxplot(x='Model', y='ROC AUC', data=df_results)
plt.title('Distribution du ROC AUC par Modèle')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('model_comparison_results.png')
print("Graphique sauvegardé sous 'model_comparison_results.png'")
plt.show()

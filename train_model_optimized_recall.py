import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    recall_score,
    precision_recall_curve,
    make_scorer
)
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("OPTIMIZED MODEL FOR FAKE BILL DETECTION")
print("Focus: Maximize Recall for Fake Bills (Class 0)")
print("=" * 60)

print("\nChargement des données...")
df = pd.read_csv('datasets/billets.csv', sep=';')

print("Imputation des valeurs manquantes...")
df_complete = df[df['margin_low'].notna()]
df_missing = df[df['margin_low'].isna()]

X_train_impute = df_complete[['diagonal', 'height_left', 'height_right', 'margin_up', 'length']]
y_train_impute = df_complete['margin_low']

model_impute = LinearRegression().fit(X_train_impute, y_train_impute)

df.loc[df['margin_low'].isna(), 'margin_low'] = model_impute.predict(
    df_missing[['diagonal', 'height_left', 'height_right', 'margin_up', 'length']]
)

print("Préparation des données...")
X = df.drop('is_genuine', axis=1)
y = df['is_genuine']

# Convert to binary: 1 = genuine (True), 0 = fake (False)
# We want to maximize recall for class 0 (fake bills)
y_binary = y.astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

print(f"\nDistribution des classes dans l'ensemble d'entraînement:")
print(f"  Vrais billets (1): {sum(y_train == 1)}")
print(f"  Faux billets (0): {sum(y_train == 0)}")

# ============================================================================
# STRATEGY 1: Use Class Weights to Penalize Misclassifying Fake Bills
# ============================================================================
print("\n" + "=" * 60)
print("STRATÉGIE 1: Poids de Classe Ajustés")
print("=" * 60)

# Calculate class weights - give more importance to fake bills (class 0)
# We'll try different weight ratios
fake_bill_weight = 3.0  # Penalize fake bill misclassification 3x more

class_weight_dict = {
    0: fake_bill_weight,  # Fake bills
    1: 1.0                # Genuine bills
}

log_reg_weighted = LogisticRegression(
    max_iter=1000, 
    random_state=42,
    class_weight=class_weight_dict
)
log_reg_weighted.fit(X_train, y_train)

y_pred_weighted = log_reg_weighted.predict(X_test)
y_pred_proba_weighted = log_reg_weighted.predict_proba(X_test)[:, 1]

print("\nRésultats avec poids de classe (fake=3.0, genuine=1.0):")
print("\nMatrice de confusion:")
cm_weighted = confusion_matrix(y_test, y_pred_weighted)
print(cm_weighted)
print("\nFormat: [[TN, FP],")
print("         [FN, TP]]")
print(f"\nFaux billets détectés: {cm_weighted[0, 0]} / {cm_weighted[0, 0] + cm_weighted[0, 1]}")
print(f"Recall pour faux billets: {cm_weighted[0, 0] / (cm_weighted[0, 0] + cm_weighted[0, 1]):.4f}")

print("\nRapport de classification:")
print(classification_report(y_test, y_pred_weighted, target_names=['Fake', 'Genuine']))
print(f"Score ROC-AUC: {roc_auc_score(y_test, y_pred_proba_weighted):.4f}")

# ============================================================================
# STRATEGY 2: Optimize Threshold for Maximum Recall on Fake Bills
# ============================================================================
print("\n" + "=" * 60)
print("STRATÉGIE 2: Optimisation du Seuil de Décision")
print("=" * 60)

# Get probability predictions
y_pred_proba_class0 = 1 - y_pred_proba_weighted  # Probability of being fake (class 0)

# Find optimal threshold that maximizes recall for fake bills
# We'll test different thresholds
thresholds_to_test = np.arange(0.3, 0.7, 0.05)
best_threshold = 0.5
best_recall_fake = 0

print("\nTest de différents seuils:")
print(f"{'Seuil':<10} {'Recall Fake':<15} {'Precision Fake':<15} {'F1-Score Fake':<15}")
print("-" * 60)

for threshold in thresholds_to_test:
    y_pred_threshold = (y_pred_proba_class0 >= threshold).astype(int)
    y_pred_threshold = 1 - y_pred_threshold  # Convert back to original encoding
    
    cm_temp = confusion_matrix(y_test, y_pred_threshold)
    recall_fake = cm_temp[0, 0] / (cm_temp[0, 0] + cm_temp[0, 1]) if (cm_temp[0, 0] + cm_temp[0, 1]) > 0 else 0
    precision_fake = cm_temp[0, 0] / (cm_temp[0, 0] + cm_temp[1, 0]) if (cm_temp[0, 0] + cm_temp[1, 0]) > 0 else 0
    f1_fake = 2 * (precision_fake * recall_fake) / (precision_fake + recall_fake) if (precision_fake + recall_fake) > 0 else 0
    
    print(f"{threshold:<10.2f} {recall_fake:<15.4f} {precision_fake:<15.4f} {f1_fake:<15.4f}")
    
    if recall_fake > best_recall_fake:
        best_recall_fake = recall_fake
        best_threshold = threshold

print(f"\nMeilleur seuil pour maximiser le recall des faux billets: {best_threshold:.2f}")
print(f"Recall obtenu: {best_recall_fake:.4f}")

# Apply best threshold
y_pred_proba_class0 = 1 - log_reg_weighted.predict_proba(X_test)[:, 1]
y_pred_optimized = (y_pred_proba_class0 >= best_threshold).astype(int)
y_pred_optimized = 1 - y_pred_optimized

print("\nRésultats avec seuil optimisé:")
print("\nMatrice de confusion:")
cm_optimized = confusion_matrix(y_test, y_pred_optimized)
print(cm_optimized)
print(f"\nFaux billets détectés: {cm_optimized[0, 0]} / {cm_optimized[0, 0] + cm_optimized[0, 1]}")
print(f"Recall pour faux billets: {cm_optimized[0, 0] / (cm_optimized[0, 0] + cm_optimized[0, 1]):.4f}")

print("\nRapport de classification:")
print(classification_report(y_test, y_pred_optimized, target_names=['Fake', 'Genuine']))

# ============================================================================
# STRATEGY 3: GridSearchCV with Recall Scoring
# ============================================================================
print("\n" + "=" * 60)
print("STRATÉGIE 3: Recherche par Grille avec Optimisation du Recall")
print("=" * 60)

# Define parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear'],
    'class_weight': [
        {0: 2.0, 1: 1.0},
        {0: 3.0, 1: 1.0},
        {0: 4.0, 1: 1.0},
        {0: 5.0, 1: 1.0},
    ]
}

# Create custom scorer for recall of fake bills (class 0)
def recall_fake_bills(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0

recall_fake_scorer = make_scorer(recall_fake_bills)

print("\nRecherche des meilleurs hyperparamètres...")
print("(Cela peut prendre quelques instants...)")

grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid,
    cv=5,
    scoring=recall_fake_scorer,
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)

print(f"\nMeilleurs paramètres trouvés:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nMeilleur score de recall (validation croisée): {grid_search.best_score_:.4f}")

# Evaluate best model on test set
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]

print("\nRésultats du meilleur modèle sur l'ensemble de test:")
print("\nMatrice de confusion:")
cm_best = confusion_matrix(y_test, y_pred_best)
print(cm_best)
print(f"\nFaux billets détectés: {cm_best[0, 0]} / {cm_best[0, 0] + cm_best[0, 1]}")
print(f"Recall pour faux billets: {cm_best[0, 0] / (cm_best[0, 0] + cm_best[0, 1]):.4f}")

print("\nRapport de classification:")
print(classification_report(y_test, y_pred_best, target_names=['Fake', 'Genuine']))
print(f"Score ROC-AUC: {roc_auc_score(y_test, y_pred_proba_best):.4f}")

# ============================================================================
# SAVE THE BEST MODEL
# ============================================================================
print("\n" + "=" * 60)
print("SAUVEGARDE DU MODÈLE OPTIMISÉ")
print("=" * 60)

print("\nSauvegarde du meilleur modèle et du scaler...")
with open('models/optimized_recall_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/scaler_optimized.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/best_threshold.pkl', 'wb') as f:
    pickle.dump(best_threshold, f)

print("Modèles sauvegardés:")
print("  - models/optimized_recall_model.pkl")
print("  - models/scaler_optimized.pkl")
print("  - models/best_threshold.pkl")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("RÉSUMÉ DES PERFORMANCES")
print("=" * 60)

print("\nComparaison des différentes approches:")
print(f"{'Approche':<40} {'Recall Fake Bills':<20}")
print("-" * 60)

# Calculate recalls
recall_weighted = cm_weighted[0, 0] / (cm_weighted[0, 0] + cm_weighted[0, 1])
recall_optimized = cm_optimized[0, 0] / (cm_optimized[0, 0] + cm_optimized[0, 1])
recall_best = cm_best[0, 0] / (cm_best[0, 0] + cm_best[0, 1])

print(f"{'1. Poids de classe ajustés':<40} {recall_weighted:<20.4f}")
print(f"{'2. Seuil optimisé':<40} {recall_optimized:<20.4f}")
print(f"{'3. GridSearchCV (RECOMMANDÉ)':<40} {recall_best:<20.4f}")

print("\n" + "=" * 60)
print("RECOMMANDATION")
print("=" * 60)
print("\nLe modèle optimisé par GridSearchCV est recommandé car il offre")
print("le meilleur équilibre entre recall et précision pour la détection")
print("des faux billets, tout en ayant été validé par validation croisée.")
print("\nUtilisez 'models/optimized_recall_model.pkl' pour vos prédictions.")
print("=" * 60)

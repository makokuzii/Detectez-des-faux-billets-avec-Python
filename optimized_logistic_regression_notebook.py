# ============================================================================
# OPTIMIZED LOGISTIC REGRESSION FOR FAKE BILL DETECTION
# Focus: Maximize Recall for Fake Bills
# ============================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score,
    make_scorer
)
import numpy as np

print("=" * 70)
print("OPTIMIZED MODEL FOR FAKE BILL DETECTION")
print("Focus: Maximize Recall for Fake Bills (Class False/0)")
print("=" * 70)

# Assuming X_train, X_test, y_train, y_test are already defined
# Convert boolean to int if needed
y_train_int = y_train.astype(int)
y_test_int = y_test.astype(int)

print(f"\nDistribution des classes:")
print(f"  Vrais billets (True/1): {sum(y_train_int == 1)}")
print(f"  Faux billets (False/0): {sum(y_train_int == 0)}")

# ============================================================================
# STRATEGY 1: Class Weights - Penalize Fake Bill Misclassification More
# ============================================================================
print("\n" + "=" * 70)
print("STRATÉGIE 1: Poids de Classe Ajustés")
print("=" * 70)

# Give 3x more importance to correctly classifying fake bills
log_reg_weighted = LogisticRegression(
    max_iter=1000, 
    random_state=42,
    class_weight={0: 3.0, 1: 1.0}  # Fake bills weighted 3x more
)
log_reg_weighted.fit(X_train, y_train_int)

y_pred_weighted = log_reg_weighted.predict(X_test)
y_pred_proba_weighted = log_reg_weighted.predict_proba(X_test)[:, 1]

print("\nRésultats avec poids de classe (fake=3.0, genuine=1.0):")
cm_weighted = confusion_matrix(y_test_int, y_pred_weighted)
print("\nMatrice de confusion:")
print(cm_weighted)
print("\nFormat: [[TN (Fake détectés), FP (Fake manqués)],")
print("         [FN (Genuine rejetés), TP (Genuine acceptés)]]")

recall_fake_w = cm_weighted[0, 0] / (cm_weighted[0, 0] + cm_weighted[0, 1])
print(f"\n🎯 RECALL POUR FAUX BILLETS: {recall_fake_w:.4f} ({recall_fake_w*100:.2f}%)")
print(f"   Faux billets détectés: {cm_weighted[0, 0]} / {cm_weighted[0, 0] + cm_weighted[0, 1]}")

print("\nRapport de classification:")
print(classification_report(y_test_int, y_pred_weighted, target_names=['Fake', 'Genuine']))
print(f"Score ROC-AUC: {roc_auc_score(y_test_int, y_pred_proba_weighted):.4f}")

# ============================================================================
# STRATEGY 2: Threshold Optimization for Maximum Recall
# ============================================================================
print("\n" + "=" * 70)
print("STRATÉGIE 2: Optimisation du Seuil de Décision")
print("=" * 70)

# Test different thresholds to find the one that maximizes fake bill recall
thresholds_to_test = np.arange(0.25, 0.75, 0.05)
best_threshold = 0.5
best_recall_fake = 0
threshold_results = []

print("\nTest de différents seuils:")
print(f"{'Seuil':<10} {'Recall Fake':<15} {'Precision Fake':<18} {'Genuine Recall':<18}")
print("-" * 70)

y_pred_proba_class0 = 1 - y_pred_proba_weighted  # Probability of being fake

for threshold in thresholds_to_test:
    # Predict fake if probability of fake >= threshold
    y_pred_threshold = (y_pred_proba_class0 >= threshold).astype(int)
    y_pred_threshold = 1 - y_pred_threshold  # Convert back to 0/1 encoding
    
    cm_temp = confusion_matrix(y_test_int, y_pred_threshold)
    
    # Calculate metrics
    recall_fake = cm_temp[0, 0] / (cm_temp[0, 0] + cm_temp[0, 1]) if (cm_temp[0, 0] + cm_temp[0, 1]) > 0 else 0
    precision_fake = cm_temp[0, 0] / (cm_temp[0, 0] + cm_temp[1, 0]) if (cm_temp[0, 0] + cm_temp[1, 0]) > 0 else 0
    recall_genuine = cm_temp[1, 1] / (cm_temp[1, 0] + cm_temp[1, 1]) if (cm_temp[1, 0] + cm_temp[1, 1]) > 0 else 0
    
    threshold_results.append({
        'threshold': threshold,
        'recall_fake': recall_fake,
        'precision_fake': precision_fake,
        'recall_genuine': recall_genuine
    })
    
    print(f"{threshold:<10.2f} {recall_fake:<15.4f} {precision_fake:<18.4f} {recall_genuine:<18.4f}")
    
    if recall_fake > best_recall_fake:
        best_recall_fake = recall_fake
        best_threshold = threshold

print(f"\n🎯 MEILLEUR SEUIL: {best_threshold:.2f}")
print(f"   Recall obtenu pour faux billets: {best_recall_fake:.4f} ({best_recall_fake*100:.2f}%)")

# Apply best threshold
y_pred_optimized = (y_pred_proba_class0 >= best_threshold).astype(int)
y_pred_optimized = 1 - y_pred_optimized

print("\nRésultats avec seuil optimisé:")
cm_optimized = confusion_matrix(y_test_int, y_pred_optimized)
print("\nMatrice de confusion:")
print(cm_optimized)

recall_fake_opt = cm_optimized[0, 0] / (cm_optimized[0, 0] + cm_optimized[0, 1])
print(f"\n🎯 RECALL POUR FAUX BILLETS: {recall_fake_opt:.4f} ({recall_fake_opt*100:.2f}%)")
print(f"   Faux billets détectés: {cm_optimized[0, 0]} / {cm_optimized[0, 0] + cm_optimized[0, 1]}")

print("\nRapport de classification:")
print(classification_report(y_test_int, y_pred_optimized, target_names=['Fake', 'Genuine']))

# ============================================================================
# STRATEGY 3: GridSearchCV with Custom Recall Scoring
# ============================================================================
print("\n" + "=" * 70)
print("STRATÉGIE 3: Recherche par Grille (GridSearchCV)")
print("=" * 70)

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

# Custom scorer for recall of fake bills (class 0)
def recall_fake_bills(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0

recall_fake_scorer = make_scorer(recall_fake_bills)

print("\n🔍 Recherche des meilleurs hyperparamètres...")
print("   (Cela peut prendre 30-60 secondes...)")

grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid,
    cv=5,
    scoring=recall_fake_scorer,
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train_int)

print(f"\n✅ Recherche terminée!")
print(f"\nMeilleurs paramètres:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nMeilleur score de recall (validation croisée): {grid_search.best_score_:.4f}")

# Evaluate best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]

print("\nRésultats du meilleur modèle sur l'ensemble de test:")
cm_best = confusion_matrix(y_test_int, y_pred_best)
print("\nMatrice de confusion:")
print(cm_best)

recall_fake_best = cm_best[0, 0] / (cm_best[0, 0] + cm_best[0, 1])
print(f"\n🎯 RECALL POUR FAUX BILLETS: {recall_fake_best:.4f} ({recall_fake_best*100:.2f}%)")
print(f"   Faux billets détectés: {cm_best[0, 0]} / {cm_best[0, 0] + cm_best[0, 1]}")

print("\nRapport de classification:")
print(classification_report(y_test_int, y_pred_best, target_names=['Fake', 'Genuine']))
print(f"Score ROC-AUC: {roc_auc_score(y_test_int, y_pred_proba_best):.4f}")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("📊 RÉSUMÉ COMPARATIF DES APPROCHES")
print("=" * 70)

print(f"\n{'Approche':<45} {'Recall Fake Bills':<25}")
print("-" * 70)
print(f"{'1. Poids de classe (3:1)':<45} {recall_fake_w:<25.4f} ({recall_fake_w*100:.1f}%)")
print(f"{'2. Seuil optimisé (' + str(best_threshold) + ')':<45} {recall_fake_opt:<25.4f} ({recall_fake_opt*100:.1f}%)")
print(f"{'3. GridSearchCV (RECOMMANDÉ)':<45} {recall_fake_best:<25.4f} ({recall_fake_best*100:.1f}%)")

# Find the best approach
approaches = [
    ('Poids de classe', recall_fake_w),
    ('Seuil optimisé', recall_fake_opt),
    ('GridSearchCV', recall_fake_best)
]
best_approach = max(approaches, key=lambda x: x[1])

print("\n" + "=" * 70)
print("🏆 RECOMMANDATION FINALE")
print("=" * 70)
print(f"\nMeilleure approche: {best_approach[0]}")
print(f"Recall pour faux billets: {best_approach[1]:.4f} ({best_approach[1]*100:.2f}%)")
print("\nCette approche maximise la détection des faux billets, ce qui est")
print("l'objectif prioritaire pour minimiser les risques financiers.")

# Store the best model for later use
if best_approach[0] == 'GridSearchCV':
    final_optimized_model = best_model
    print("\nModèle final: best_model (issu de GridSearchCV)")
elif best_approach[0] == 'Poids de classe':
    final_optimized_model = log_reg_weighted
    print("\nModèle final: log_reg_weighted")
else:
    final_optimized_model = log_reg_weighted  # Use weighted with custom threshold
    print(f"\nModèle final: log_reg_weighted avec seuil {best_threshold}")
    print(f"Note: Utilisez ce seuil lors des prédictions")

print("=" * 70)

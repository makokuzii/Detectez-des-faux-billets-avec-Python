import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("SVM - Détection de Faux Billets")
print("=" * 60)

# Chargement des données
print("\n[1/6] Chargement des données...")
df = pd.read_csv('datasets/billets.csv', sep=';')
print(f"   → {len(df)} billets chargés")
print(f"   → Valeurs manquantes dans 'margin_low': {df['margin_low'].isna().sum()}")

# Imputation des valeurs manquantes
print("\n[2/6] Imputation des valeurs manquantes...")
df_complete = df[df['margin_low'].notna()]
df_missing = df[df['margin_low'].isna()]

X_train_impute = df_complete[['diagonal', 'height_left', 'height_right', 'margin_up', 'length']]
y_train_impute = df_complete['margin_low']

model_impute = LinearRegression().fit(X_train_impute, y_train_impute)

df.loc[df['margin_low'].isna(), 'margin_low'] = model_impute.predict(
    df_missing[['diagonal', 'height_left', 'height_right', 'margin_up', 'length']]
)
print(f"   → {len(df_missing)} valeurs imputées avec succès")

# Préparation des données
print("\n[3/6] Préparation des données...")
X = df.drop('is_genuine', axis=1)
y = df['is_genuine']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
print(f"   → Ensemble d'entraînement: {len(X_train)} échantillons")
print(f"   → Ensemble de test: {len(X_test)} échantillons")

# Recherche des meilleurs hyperparamètres avec GridSearchCV
print("\n[4/6] Optimisation des hyperparamètres (GridSearchCV)...")
print("   → Cela peut prendre quelques minutes...")

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear', 'poly']
}

svm_model = SVC(random_state=42, probability=True)

grid_search = GridSearchCV(
    svm_model,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\n   ✓ Meilleurs paramètres trouvés:")
for param, value in grid_search.best_params_.items():
    print(f"      • {param}: {value}")
print(f"   ✓ Meilleur score de validation croisée: {grid_search.best_score_:.4f}")

# Entraînement du modèle final avec les meilleurs paramètres
print("\n[5/6] Entraînement du modèle SVM final...")
best_svm = grid_search.best_estimator_

# Validation croisée
cv_scores = cross_val_score(best_svm, X_train, y_train, cv=5, scoring='accuracy')
print(f"   → Scores de validation croisée: {cv_scores}")
print(f"   → Score moyen: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Prédictions
y_pred = best_svm.predict(X_test)
y_pred_proba = best_svm.predict_proba(X_test)[:, 1]

# Évaluation
print("\n[6/6] Évaluation du modèle...")
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n{'=' * 60}")
print("RÉSULTATS FINAUX")
print(f"{'=' * 60}")
print(f"Précision (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"ROC-AUC Score: {roc_auc:.4f}")

print("\nMatrice de Confusion:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\n   Vrais Négatifs (TN): {cm[0][0]}")
print(f"   Faux Positifs (FP): {cm[0][1]}")
print(f"   Faux Négatifs (FN): {cm[1][0]}")
print(f"   Vrais Positifs (TP): {cm[1][1]}")

print("\nRapport de Classification:")
print(classification_report(y_test, y_pred, target_names=['Faux', 'Vrai']))

# Visualisations
print("\n[BONUS] Génération des visualisations...")

# 1. Matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Faux', 'Vrai'], 
            yticklabels=['Faux', 'Vrai'])
plt.title('Matrice de Confusion - SVM')
plt.ylabel('Valeur Réelle')
plt.xlabel('Prédiction')
plt.tight_layout()
plt.savefig('svm_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("   ✓ Matrice de confusion sauvegardée: svm_confusion_matrix.png")

# 2. Courbe ROC
plt.figure(figsize=(8, 6))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Hasard')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC - SVM')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('svm_roc_curve.png', dpi=300, bbox_inches='tight')
print("   ✓ Courbe ROC sauvegardée: svm_roc_curve.png")

# 3. Comparaison des scores de validation croisée
plt.figure(figsize=(10, 6))
plt.bar(range(1, 6), cv_scores, color='steelblue', alpha=0.7, edgecolor='black')
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', 
            label=f'Moyenne: {cv_scores.mean():.4f}')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Scores de Validation Croisée (5-Fold)')
plt.ylim([0.95, 1.0])
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('svm_cv_scores.png', dpi=300, bbox_inches='tight')
print("   ✓ Scores CV sauvegardés: svm_cv_scores.png")

# Sauvegarde du modèle
print("\n[SAUVEGARDE] Enregistrement du modèle et du scaler...")
with open('models/svm_model.pkl', 'wb') as f:
    pickle.dump(best_svm, f)
with open('models/svm_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("   ✓ Modèle sauvegardé: models/svm_model.pkl")
print("   ✓ Scaler sauvegardé: models/svm_scaler.pkl")

print(f"\n{'=' * 60}")
print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS ✓")
print(f"{'=' * 60}\n")

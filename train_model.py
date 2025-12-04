import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

print("Chargement des données...")
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
y = df['is_genuine'].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print("Entraînement du modèle de régression logistique...")
final_model = LogisticRegression(random_state=42, class_weight={0: 3, 1: 1}, max_iter=100)
final_model.fit(X_train, y_train)

print("Sauvegarde du modèle et du scaler...")
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Terminé.")
print(f"Précision du modèle sur l'ensemble de test : {final_model.score(X_test, y_test):.4f}")

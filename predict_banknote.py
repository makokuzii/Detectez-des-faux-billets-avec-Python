import argparse
import pickle
import pandas as pd
import numpy as np
import sys
import os

# Constantes
MODEL_PATH = 'models/best_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
# Ordre des caractéristiques attendu par le scaler/modèle (basé sur le script d'entraînement)
# Dans l'entraînement : X = df.drop('is_genuine', axis=1)
# Ordre CSV : is_genuine;diagonal;height_left;height_right;margin_low;margin_up;length
# Donc les colonnes X sont : diagonal, height_left, height_right, margin_low, margin_up, length
FEATURE_COLUMNS = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']

def load_artifacts():
    """Charger le modèle entraîné et le scaler."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Erreur : Fichiers du modèle introuvables.")
        print(f"Attendu : {MODEL_PATH} et {SCALER_PATH}")
        print("Veuillez d'abord exécuter train_model.py.")
        sys.exit(1)
        
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def get_input_interactive():
    """Demander à l'utilisateur de saisir les valeurs."""
    print("\n--- Authentificateur de Billets ---")
    print("Veuillez entrer les dimensions du billet (en mm) :")
    
    data = {}
    try:
        data['diagonal'] = float(input("Diagonal: "))
        data['height_left'] = float(input("Height Left: "))
        data['height_right'] = float(input("Height Right: "))
        data['margin_low'] = float(input("Margin Low: "))
        data['margin_up'] = float(input("Margin Up: "))
        data['length'] = float(input("Length: "))
    except ValueError:
        print("Erreur : Entrée invalide. Veuillez entrer des valeurs numériques.")
        sys.exit(1)
        
    return data

def predict(model, scaler, data):
    """Faire une prédiction basée sur les données d'entrée."""
    # Créer un DataFrame avec l'ordre correct des colonnes
    df = pd.DataFrame([data], columns=FEATURE_COLUMNS)
    
    # Normaliser les caractéristiques
    X_scaled = scaler.transform(df)
    X_scaled = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)
    
    # Prédire
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    prob_fake = probabilities[0]      # probability of Fake (class 0)
    prob_genuine = probabilities[1]   # probability of Genuine (class 1)
    
    return prediction, prob_fake, prob_genuine

def main():
    parser = argparse.ArgumentParser(description="Prédire si un billet est Vrai ou Faux.")
    
    # Optional arguments
    parser.add_argument('--diagonal', type=float, help='Diagonal length')
    parser.add_argument('--height_left', type=float, help='Height left')
    parser.add_argument('--height_right', type=float, help='Height right')
    parser.add_argument('--margin_low', type=float, help='Margin low')
    parser.add_argument('--margin_up', type=float, help='Margin up')
    parser.add_argument('--length', type=float, help='Length')
    
    args = parser.parse_args()
    
    # Vérifier si tous les arguments sont fournis via CLI
    if all(v is not None for v in [args.diagonal, args.height_left, args.height_right, args.margin_low, args.margin_up, args.length]):
        data = {
            'diagonal': args.diagonal,
            'height_left': args.height_left,
            'height_right': args.height_right,
            'margin_low': args.margin_low,
            'margin_up': args.margin_up,
            'length': args.length
        }
    else:
        # Si tous les arguments ne sont pas fournis, passer en mode interactif
        data = get_input_interactive()

    model, scaler = load_artifacts()
    is_genuine, prob_fake, prob_genuine = predict(model, scaler, data)
    
    result_text = "Vrai Billet" if is_genuine else "Faux Billet"
    color_code = "\033[92m" if is_genuine else "\033[91m" # Vert pour Vrai, Rouge pour Faux
    reset_code = "\033[0m"
    
    print(f"\nPrédiction : {color_code}{result_text}{reset_code}")
    print(f"Probabilité d'être Vrai : {prob_genuine:.2%}")
    print(f"Probabilité d'être Faux : {prob_fake:.2%}")

if __name__ == "__main__":
    main()
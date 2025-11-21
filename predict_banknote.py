import argparse
import pickle
import pandas as pd
import numpy as np
import sys
import os

# Constants
MODEL_PATH = 'models/best_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
# Feature order expected by the scaler/model (based on training script)
# In training: X = df.drop('is_genuine', axis=1)
# CSV order: is_genuine;diagonal;height_left;height_right;margin_low;margin_up;length
# So X columns are: diagonal, height_left, height_right, margin_low, margin_up, length
FEATURE_COLUMNS = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']

def load_artifacts():
    """Load the trained model and scaler."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: Model files not found.")
        print(f"Expected: {MODEL_PATH} and {SCALER_PATH}")
        print("Please run train_model.py first.")
        sys.exit(1)
        
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def get_input_interactive():
    """Prompt user for input values."""
    print("\n--- Banknote Authenticator ---")
    print("Please enter the banknote dimensions (in mm):")
    
    data = {}
    try:
        data['diagonal'] = float(input("Diagonal: "))
        data['height_left'] = float(input("Height Left: "))
        data['height_right'] = float(input("Height Right: "))
        data['margin_low'] = float(input("Margin Low: "))
        data['margin_up'] = float(input("Margin Up: "))
        data['length'] = float(input("Length: "))
    except ValueError:
        print("Error: Invalid input. Please enter numeric values.")
        sys.exit(1)
        
    return data

def predict(model, scaler, data):
    """Make a prediction based on input data."""
    # Create DataFrame with correct column order
    df = pd.DataFrame([data], columns=FEATURE_COLUMNS)
    
    # Scale features
    X_scaled = scaler.transform(df)
    X_scaled = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    # Probability of True (Genuine)
    prob_genuine = probability[1]
    
    return prediction, prob_genuine

def main():
    parser = argparse.ArgumentParser(description="Predict if a banknote is Real or Fake.")
    
    # Optional arguments
    parser.add_argument('--diagonal', type=float, help='Diagonal length')
    parser.add_argument('--height_left', type=float, help='Height left')
    parser.add_argument('--height_right', type=float, help='Height right')
    parser.add_argument('--margin_low', type=float, help='Margin low')
    parser.add_argument('--margin_up', type=float, help='Margin up')
    parser.add_argument('--length', type=float, help='Length')
    
    args = parser.parse_args()
    
    # Check if all args are provided via CLI
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
        # If not all args provided, switch to interactive
        data = get_input_interactive()

    model, scaler = load_artifacts()
    is_genuine, prob = predict(model, scaler, data)
    
    result_text = "Real Banknote" if is_genuine else "Fake Banknote"
    color_code = "\033[92m" if is_genuine else "\033[91m" # Green for Real, Red for Fake
    reset_code = "\033[0m"
    
    print(f"\nPrediction: {color_code}{result_text}{reset_code}")
    print(f"Probability of being Real: {prob:.2%}")

if __name__ == "__main__":
    main()

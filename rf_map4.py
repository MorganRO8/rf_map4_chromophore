import os
import optuna
import hashlib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
from rdkit import Chem
from map4 import MAP4Calculator
import joblib

# Function to convert SMILES to MAP4 fingerprint
def smiles_to_map4(smiles_list, n_bits=1024):
    map4_calculator = MAP4Calculator(dimensions=n_bits)
    map4_fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            map4 = map4_calculator.calculate(mol)
            map4_fps.append(map4)
        else:
            map4_fps.append([0]*n_bits)
    return pd.DataFrame(map4_fps)

# Objective function for Optuna to optimize
def objective(trial, X_train, y_train, X_val, y_val):

    max_features_type = trial.suggest_categorical('max_features_type', ['int', 'float', 'str', 'None'])
    if max_features_type == 'int':
        max_features = trial.suggest_int('max_features_int', 1, X_train.shape[1])
    elif max_features_type == 'float':
        max_features = trial.suggest_float('max_features_float', 0.1, 1.0)
    elif max_features_type == 'str':
        max_features = trial.suggest_categorical('max_features_str', ['sqrt', 'log2'])
    else:
        max_features = None

    n_estimators = trial.suggest_int('n_estimators', 2, 512)
    max_depth = trial.suggest_int('max_depth', 1, 256)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 14)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 14)
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    criterion = trial.suggest_categorical('criterion', ['friedman_mse', 'absolute_error', 'squared_error', 'poisson'])
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 256)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        criterion=criterion,
        max_leaf_nodes=max_leaf_nodes,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return mean_absolute_error(y_val, y_pred)

# Load the data
file_path = 'Fluorescent_Molecules_Database.csv'
df = pd.read_csv(file_path, skiprows=1)
df.columns = ['Tag', 'Chromophore (SMILES)', 'Absorption max (nm)', 'Emission max (nm)', 'Lifetime (ns)', 'Quantum yield',
              'log(e/mol-1 dm3 cm-1)', 'abs FWHM (cm-1)', 'emi FWHM (cm-1)', 'abs FWHM (nm)', 'emi FWHM (nm)',
              'Molecular weight (g mol-1)', 'Reference']

# Convert numerical columns to appropriate types
numerical_columns = ['Absorption max (nm)', 'Emission max (nm)', 'Lifetime (ns)', 'Quantum yield']
df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')

# Remove rows where at least one target value is missing
df = df.dropna(subset=['Absorption max (nm)', 'Emission max (nm)', 'Quantum yield'])

# Convert SMILES to MAP4 fingerprints
smiles_list = df['Chromophore (SMILES)'].tolist()
X = smiles_to_map4(smiles_list)

# Set target columns
y_targets = {
    'Absorption max': df['Absorption max (nm)'],
    'Emission max': df['Emission max (nm)'],
    'Quantum yield': df['Quantum yield']
}

# Create folders if they don't exist
for target in y_targets.keys():
    folder_path = os.path.join(os.getcwd(), target)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Initialize variables to keep track of best models
best_scores = {
    'Absorption max': float('inf'),
    'Emission max': float('inf'),
    'Quantum yield': float('inf')
}

# Continuous training until interrupted
try:
    while True:
        # Split and train data for each target
        for target, y in y_targets.items():
            folder_path = os.path.join(os.getcwd(), target)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            # Optuna optimization
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=20)
            
            # Train model with best parameters
            trial = study.best_trial
            model = RandomForestRegressor(n_estimators=trial.params['n_estimators'], max_depth=trial.params['max_depth'])
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            print(f"Mean Absolute Error for {target}: {mae}")

            # Check for improvement and save model
            if mae < best_scores[target] * 0.99:  # 1% improvement
                best_scores[target] = mae
                model_name = f"{int(100000/mae)}_{target}_{trial.params['n_estimators']}est_{trial.params['max_depth']}depth.pkl"
                model_path = os.path.join(folder_path, model_name)
                joblib.dump(model, model_path)
                print(f"Saved new best model for {target} with MAE: {mae}")

except:
    print("Training interrupted. Exiting.")

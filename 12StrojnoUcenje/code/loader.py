import pandas as pd
import os

def load_data_from_path(h5_file_path):
    """Load HIGGS data directly from .h5 file path"""
    print("Loading {}...".format(h5_file_path))
    dataset = pd.HDFStore(h5_file_path, "r")
    print("Loaded.")
    return dataset

def inspect_h5_file(h5_file_path):
    """Inspect what's inside an H5 file"""
    print(f"=== Inspecting {h5_file_path} ===")
    
    if not os.path.exists(h5_file_path):
        print(f"File not found: {h5_file_path}")
        return
    
    try:
        with pd.HDFStore(h5_file_path, 'r') as store:
            print(f"Keys in file: {store.keys()}")
            
            if '/train' in store:
                train_shape = store['train'].shape
                print(f"Training data shape: {train_shape}")
            
            if '/valid' in store:
                val_shape = store['valid'].shape  
                print(f"Validation data shape: {val_shape}")
                
            if '/feature_names' in store:
                features = store['feature_names']
                print(f"Number of features: {len(features)}")
                print(f"Feature names: {list(features)}")
            else:
                print("No feature names found in file")
                
    except Exception as e:
        print(f"Error reading file: {e}")

def load_higgs_dataset(h5_file_path, return_format='dataframes'):
    """
    Load HIGGS dataset with different return formats
    
    Args:
        h5_file_path: Path to the .h5 file
        return_format: 'store', 'dataframes', 'arrays', or 'split'
    
    Returns:
        Depends on return_format:
        - 'store': Returns the HDFStore object (you must close it!)
        - 'dataframes': Returns (train_df, val_df, feature_names)
        - 'arrays': Returns (train_array, val_array, feature_names) 
        - 'split': Returns (X_train, y_train, X_val, y_val, feature_names)
    """
    
    if not os.path.exists(h5_file_path):
        raise FileNotFoundError(f"File not found: {h5_file_path}")
    
    print(f"Loading {h5_file_path}...")
    
    if return_format == 'store':
        # Return HDFStore object (user must close it)
        dataset = pd.HDFStore(h5_file_path, "r")
        print("Loaded as HDFStore object. Remember to close it!")
        return dataset
    
    # For other formats, load data and close store
    with pd.HDFStore(h5_file_path, 'r') as store:
        train_df = store['train'].copy()
        val_df = store['valid'].copy()
        
        if 'feature_names' in store:
            feature_names = store['feature_names'].copy()
        else:
            # Fallback: use column names from dataframe
            feature_names = pd.Series(train_df.columns)
    
    print("Loaded successfully!")
    print(f"Training data shape: {train_df.shape}")
    print(f"Validation data shape: {val_df.shape}")
    print(f"Features: {list(feature_names)}")
    
    if return_format == 'dataframes':
        return train_df, val_df, feature_names
    
    elif return_format == 'arrays':
        return train_df.values, val_df.values, feature_names
    
    elif return_format == 'split':
        # Split features from labels
        X_train = train_df.drop(['hlabel'], axis=1)
        y_train = train_df['hlabel']
        X_val = val_df.drop(['hlabel'], axis=1)
        y_val = val_df['hlabel']
        
        # Remove 'hlabel' from feature names
        feature_names_no_label = feature_names[feature_names != 'hlabel']
        
        return X_train, y_train, X_val, y_val, feature_names_no_label
    
    else:
        raise ValueError(f"Unknown return_format: {return_format}. Use 'store', 'dataframes', 'arrays', or 'split'")

def split_xy(dataframe):
    """Split dataframe into features (X) and labels (y) - matches your existing code"""
    X = dataframe.drop(['hlabel'], axis=1)
    y = dataframe['hlabel']
    return X, y

def load_and_split(h5_file_path):
    """
    Load data and return in the format matching your existing code:
    x_trn, y_trn = split_xy(hdata['train'])
    x_val, y_val = split_xy(hdata['valid'])
    """
    
    # Load using your exact function
    hdata = load_data_from_path(h5_file_path)
    
    # Extract data
    data_fnames = hdata['feature_names'].to_numpy()[1:]  # drop labels
    n_dims = data_fnames.shape[0]
    print(f"Entries read {n_dims} with feature names {data_fnames}")
    
    # Split features and labels
    x_trn, y_trn = split_xy(hdata['train'])
    x_val, y_val = split_xy(hdata['valid'])
    
    # Close the store
    hdata.close()
    
    print(f"Training shape: {x_trn.shape}, Validation shape: {x_val.shape}")
    
    return x_trn, y_trn, x_val, y_val, data_fnames

# def main():
#     """Example usage of the loader"""
    
#     # Example file paths (adjust to your actual files)
#     files_to_try = [
#         '12StrojnoUcenje/code/data/higgs-parsed.h5'
#     ]
    
#     for file_path in files_to_try:
#         if os.path.exists(file_path):
#             print(f"\n=== Loading {file_path} ===")
            
#             # Method 1: Your exact approach
#             print("\n--- Using your original approach ---")
#             try:
#                 x_trn, y_trn, x_val, y_val, feature_names = load_and_split(file_path)
#                 print(f"Success! Features: {feature_names}")
#             except Exception as e:
#                 print(f"Error: {e}")
            
#             # Method 2: Alternative approaches
#             print("\n--- Alternative loading methods ---")
#             try:
#                 # Load as split data ready for ML
#                 X_train, y_train, X_val, y_val, features = load_higgs_dataset(file_path, 'split')
#                 print(f"Split format - X_train shape: {X_train.shape}")
                
#                 # Check class balance
#                 print(f"Training - Signal: {(y_train == 1).sum()}, Background: {(y_train == 0).sum()}")
#                 print(f"Validation - Signal: {(y_val == 1).sum()}, Background: {(y_val == 0).sum()}")
                
#             except Exception as e:
#                 print(f"Error with alternative method: {e}")
#         else:
#             print(f"File not found: {file_path}")

# if __name__ == "__main__":
#     # Inspect a file
#     # inspect_h5_file('./data/higgs_all_features.h5')
    
#     # Run examples
#     main()
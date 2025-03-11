def save_processed_data(data, file_path):
    """
    Save processed data to file for easy loading later
    
    Parameters:
    data: Data to save (DataFrame or dictionary of DataFrames)
    file_path (str): Path to save the file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save data using pickle for preserving DataFrame structure
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Successfully saved processed data to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

def load_processed_data(file_path):
    """
    Load previously processed data from file
    
    Parameters:
    file_path (str): Path to the saved data file
    
    Returns:
    The loaded data (DataFrame or dictionary of DataFrames)
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Successfully loaded processed data from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

import numpy as np
import pandas as pd
import os
import re
import pickle

def load_gas_sensor_batch(batch_file):
    """
    Load a batch file in modified LIBSVM format with concentration levels
    and convert to pandas DataFrame
    
    Parameters:
    batch_file (str): Path to the batch file
    
    Returns:
    DataFrame: DataFrame with features, class, and concentration
    """
    # Lists to store data
    X_data = []
    classes = []
    concentrations = []
    
    # Read the file line by line
    with open(batch_file, 'r') as f:
        for line in f:
            # Parse each line
            parts = line.strip().split()
            
            # Extract class and concentration (formatted as "class;concentration")
            if ';' in parts[0]:
                class_conc = parts[0].split(';')
                gas_class = int(class_conc[0])
                concentration = float(class_conc[1])
            else:
                # If no concentration, assume it's just the class
                gas_class = int(parts[0])
                concentration = None
            
            # Extract features
            features = {}
            for feature_val in parts[1:]:
                if ':' in feature_val:
                    idx, val = feature_val.split(':')
                    features[int(idx)] = float(val)
            
            # Ensure all 128 features exist
            feature_vector = [features.get(i, 0.0) for i in range(1, 129)]
            
            # Add to our lists
            X_data.append(feature_vector)
            classes.append(gas_class)
            concentrations.append(concentration)
    
    # Create feature column names
    feature_names = []
    for sensor_id in range(1, 17):  # 16 sensors
        # 8 features per sensor as described in the documentation
        feature_names.append(f"ΔR_{sensor_id}")
        feature_names.append(f"|ΔR|_{sensor_id}")
        feature_names.append(f"EMAi0.001_{sensor_id}")
        feature_names.append(f"EMAi0.01_{sensor_id}")
        feature_names.append(f"EMAi0.1_{sensor_id}")
        feature_names.append(f"EMAd0.001_{sensor_id}")
        feature_names.append(f"EMAd0.01_{sensor_id}")
        feature_names.append(f"EMAd0.1_{sensor_id}")
    
    # Map class labels to gas names
    class_mapping = {
        1: "Ethanol",
        2: "Ethylene",
        3: "Ammonia",
        4: "Acetaldehyde",
        5: "Acetone",
        6: "Toluene"
    }
    
    # Create DataFrame
    df = pd.DataFrame(X_data, columns=feature_names)
    df['gas_class'] = classes
    df['gas_name'] = df['gas_class'].map(class_mapping)
    df['concentration_ppmv'] = concentrations
    
    return df

def extract_all_batches(data_dir):
    """
    Extract all 10 batches from the dataset
    
    Parameters:
    data_dir (str): Directory containing the batch files
    
    Returns:
    dict: Dictionary of DataFrames, one for each batch
    """
    batches = {}
    
    for i in range(1, 11):
        batch_file = os.path.join(data_dir, f"batch{i}.dat")
        if os.path.exists(batch_file):
            df = load_gas_sensor_batch(batch_file)
            batches[f"Batch{i}"] = df
            print(f"Loaded Batch {i} with {len(df)} samples")
        else:
            print(f"Warning: Batch {i} file not found at {batch_file}")
    
    return batches

def extract_and_merge_all_batches(data_dir):
    """
    Extract all batches and merge them into one DataFrame with batch information
    
    Parameters:
    data_dir (str): Directory containing the batch files
    
    Returns:
    DataFrame: Combined DataFrame with all batches
    """
    all_dfs = []
    
    for i in range(1, 11):
        batch_file = os.path.join(data_dir, f"batch{i}.dat")
        if os.path.exists(batch_file):
            df = load_gas_sensor_batch(batch_file)
            df['batch_id'] = i
            # Add month information based on dataset documentation
            if i == 1:
                months = [1, 2]
            elif i == 2:
                months = [3, 4, 8, 9, 10]
            elif i == 3:
                months = [11, 12, 13]
            elif i == 4:
                months = [14, 15]
            elif i == 5:
                months = [16]
            elif i == 6:
                months = [17, 18, 19, 20]
            elif i == 7:
                months = [21]
            elif i == 8:
                months = [22, 23]
            elif i == 9:
                months = [24, 30]
            elif i == 10:
                months = [36]
            else:
                months = []
                
            df['months'] = str(months)
            all_dfs.append(df)
            print(f"Loaded Batch {i} with {len(df)} samples from months {months}")
        else:
            print(f"Warning: Batch {i} file not found at {batch_file}")
    
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Set this to the directory containing your batch files
    os.chdir("ESE-05-Supervised machine learning")
    data_directory = "dataset"
    
    # Set paths for saving processed data
    individual_batches_path = "./processed_data/individual_batches.pkl"
    merged_batches_path = "./processed_data/merged_batches.pkl"
    
    # Check if processed data already exists
    if os.path.exists(merged_batches_path):
        print(f"Loading previously processed merged data from {merged_batches_path}")
        merged_df = load_processed_data(merged_batches_path)
        
        if merged_df is not None:
            print(f"Loaded merged data with shape: {merged_df.shape}")
            
            # Example analysis of the merged data
            print("\nGas distribution across all batches:")
            print(merged_df.groupby(['batch_id', 'gas_name']).size().unstack().fillna(0).astype(int))
            
            print("\nConcentration levels by gas type:")
            print(merged_df.groupby('gas_name')['concentration_ppmv'].value_counts().sort_index())
    else:
        # Extract and merge all batches
        print("\nExtracting and merging all batches...")
        merged_df = extract_and_merge_all_batches(data_directory)
        
        if not merged_df.empty:
            print(f"Merged data shape: {merged_df.shape}")
            
            # Example analysis of the merged data
            print("\nGas distribution across all batches:")
            print(merged_df.groupby(['batch_id', 'gas_name']).size().unstack().fillna(0).astype(int))
            
            print("\nConcentration levels by gas type:")
            print(merged_df.groupby('gas_name')['concentration_ppmv'].value_counts().sort_index())
            
            # Save the processed data
            print("\nSaving processed merged data for future use...")
            save_processed_data(merged_df, merged_batches_path)
    
    # Check if individual batches data exists
    if os.path.exists(individual_batches_path):
        print(f"\nLoading previously processed individual batches from {individual_batches_path}")
        all_batches = load_processed_data(individual_batches_path)
        
        if all_batches is not None:
            # Print summary of each batch
            for batch_name, batch_df in all_batches.items():
                print(f"\n{batch_name} Summary:")
                print(f"Shape: {batch_df.shape}")
                
                print("Gas distribution:")
                print(batch_df['gas_name'].value_counts())
                
                print("Concentration distribution:")
                print(batch_df['concentration_ppmv'].value_counts().sort_index())
    else:
        # Extract all batches individually
        print("\nExtracting all batches individually...")
        all_batches = extract_all_batches(data_directory)
        
        if all_batches:
            # Print summary of each batch
            for batch_name, batch_df in all_batches.items():
                print(f"\n{batch_name} Summary:")
                print(f"Shape: {batch_df.shape}")
                
                print("Gas distribution:")
                print(batch_df['gas_name'].value_counts())
                
                print("Concentration distribution:")
                print(batch_df['concentration_ppmv'].value_counts().sort_index())
            
            # Save the processed data
            print("\nSaving processed individual batches for future use...")
            save_processed_data(all_batches, individual_batches_path)
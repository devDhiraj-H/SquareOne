"""
experiments/utils/data_loader.py

Leakage-free dataset loader for UAV Cyber-Physical Attack Detection.
Harmonizes both Physical and Cyber datasets to evaluate all 5 attack classes:
['Benign', 'DoS', 'Replay', 'Evil_Twin', 'FDI']
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Standard 5-class canonical taxonomy
CANONICAL_CLASSES = ['Benign', 'DoS', 'Replay', 'Evil_Twin', 'FDI']

# Physical leakage columns to drop
PHYSICAL_LEAKAGE_COLS = [
    'timestamp_p', 'time', 'Timestamp', 
    'flight_time', 'battery',          # Monotonic clocks
    'barometer', 'temperature',        # Weather drift
    'distance', 'mp_distance_x'        # Absolute GPS/location markers
]

# Cyber leakage columns to drop
CYBER_LEAKAGE_COLS = [
    'timestamp_c', 'time', 'frame.number', 'packet_id', # Clocks / packet counters
    'wlan.sa', 'wlan.ta', 'wlan.da', 'wlan.ra',         # MAC addresses
    'ip.src', 'ip.dst', 'ip.id',                         # IP addresses
    'Unnamed: 38', 'class'                              # Label columns
]


def load_physical_dataset(csv_path="../Physical_UAV_Dataset.csv"):
    """
    Loads and cleans the Physical UAV Dataset.
    Returns:
        X (pd.DataFrame): Leakage-free physical sensor features
        y (pd.Series): Canonical target labels (5 classes)
        feature_names (list): List of feature column names
    """
    if not os.path.exists(csv_path):
        # Fallback to local path if called from root
        csv_path = "Physical_UAV_Dataset.csv"
    
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Target column
    label_col = 'Target_Label' if 'Target_Label' in df.columns else 'class'
    y = df[label_col].astype(str)
    
    # Map any variations to canonical names
    label_map = {
        'benign': 'Benign',
        'Benign': 'Benign',
        'DoS': 'DoS',
        'DoS attack': 'DoS',
        'Replay': 'Replay',
        'Evil_Twin': 'Evil_Twin',
        'FDI': 'FDI'
    }
    y = y.map(label_map).fillna(y)
    
    # Drop labels and leakage
    drop_cols = [c for c in PHYSICAL_LEAKAGE_COLS + [label_col] if c in df.columns]
    X = df.drop(columns=drop_cols)
    
    # Force all sensor readings to numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    
    return X, y, list(X.columns)


def load_cyber_dataset(csv_path="../Cyber_UAV_Dataset.csv"):
    """
    Loads and cleans the Cyber UAV Dataset, fully restoring the 5-class target labels
    from 'Unnamed: 38' and 'class' columns.
    Returns:
        X (pd.DataFrame): Leakage-free cyber network features
        y (pd.Series): Canonical target labels (5 classes)
        feature_names (list): List of feature column names
    """
    if not os.path.exists(csv_path):
        csv_path = "Cyber_UAV_Dataset.csv"
        
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Drop corrupted repeated header rows if present
    df = df[df['class'] != 'class']
    
    # Merge 'Unnamed: 38' and 'class' to restore all 5 classes
    # 'Unnamed: 38' contains Replay, DoS, Evil_Twin, FDI
    # 'class' contains benign, DoS attack, Replay
    if 'Unnamed: 38' in df.columns:
        y_raw = df['Unnamed: 38'].fillna(df['class'])
    else:
        y_raw = df['class']
        
    label_map = {
        'benign': 'Benign',
        'Benign': 'Benign',
        'DoS': 'DoS',
        'DoS attack': 'DoS',
        'Replay': 'Replay',
        'Evil_Twin': 'Evil_Twin',
        'FDI': 'FDI'
    }
    y = y_raw.map(label_map)
    
    # Drop any rows where label could not be resolved (if any)
    valid_idx = y.notna()
    df = df[valid_idx]
    y = y[valid_idx]
    
    # Drop leakage and label columns
    drop_cols = [c for c in CYBER_LEAKAGE_COLS if c in df.columns]
    X = df.drop(columns=drop_cols)
    
    # Force all packet features to numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    
    return X, y, list(X.columns)


def get_stratified_split(X, y, test_size=0.3, random_state=42):
    """
    Encodes labels and performs a stratified train/test split.
    """
    encoder = LabelEncoder()
    # Fit canonical order if present
    classes_present = sorted(list(y.unique()))
    encoder.fit(classes_present)
    y_encoded = encoder.transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test, encoder


def load_multimodal_dataset(phys_csv="../Physical_UAV_Dataset.csv", cyb_csv="../Cyber_UAV_Dataset.csv", random_state=42):
    """
    Loads and aligns Physical and Cyber datasets across the 5 canonical classes.
    Balanced alignment: samples min(N_phys, N_cyb) from each class.
    Returns:
        X_fused (pd.DataFrame): 37 fused features (9 Physical + 28 Cyber)
        y_fused (pd.Series): Canonical target labels
        feature_names (list): Combined feature column names
    """
    X_p, y_p, _ = load_physical_dataset(phys_csv)
    X_c, y_c, _ = load_cyber_dataset(cyb_csv)
    
    p_list, c_list, y_list = [], [], []
    for c in CANONICAL_CLASSES:
        n = min(sum(y_p == c), sum(y_c == c))
        p_chunk = X_p[y_p == c].sample(n=n, random_state=random_state).reset_index(drop=True)
        c_chunk = X_c[y_c == c].sample(n=n, random_state=random_state).reset_index(drop=True)
        
        p_chunk.columns = [f"phys_{col}" for col in p_chunk.columns]
        c_chunk.columns = [f"cyb_{col}" for col in c_chunk.columns]
        
        merged_chunk = pd.concat([p_chunk, c_chunk], axis=1)
        p_list.append(merged_chunk)
        y_list.extend([c] * n)
        
    X_fused = pd.concat(p_list, ignore_index=True)
    y_fused = pd.Series(y_list)
    return X_fused, y_fused, list(X_fused.columns)


def get_novelty_detection_split(X, y, benign_train_ratio=0.7, random_state=42):
    """
    Splits dataset for Unsupervised Anomaly / Novelty Detection:
    - X_train_benign: Benign samples only (UAV normal baseline telemetry)
    - X_test: Unseen Benign test samples + all Attack samples
    - y_test_binary: 0 for Benign, 1 for Attack (numpy array)
    - y_test_multiclass: Original string labels ('Benign', 'DoS', 'Evil_Twin', 'FDI', 'Replay')
    """
    benign_mask = (y == 'Benign')
    X_benign = X[benign_mask]
    y_benign = y[benign_mask]
    
    X_attack = X[~benign_mask]
    y_attack = y[~benign_mask]
    
    X_train_benign, X_test_benign, _, y_test_benign = train_test_split(
        X_benign, y_benign, train_size=benign_train_ratio, random_state=random_state
    )
    
    X_test = pd.concat([X_test_benign, X_attack], ignore_index=True)
    y_test_multiclass = pd.concat([y_test_benign, y_attack], ignore_index=True)
    y_test_binary = (y_test_multiclass != 'Benign').astype(int).values
    
    return X_train_benign, X_test, y_test_binary, y_test_multiclass


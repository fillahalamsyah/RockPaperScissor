# dataset.py
# dataset.py berisi kode untuk memuat dataset yang digunakan dalam permainan dan model.

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Variabel Global
choices = ['rock', 'paper', 'scissors']

# mendapatkan path file ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Path dataset
dataset_path = os.path.join(BASE_DIR, "augmented_dataset.csv")

# Dataset Awal
data = [('rock', 'scissors', 1), ('rock', 'paper', 0), ('rock', 'rock', 0.5),
        ('paper', 'rock', 1), ('paper', 'scissors', 0), ('paper', 'paper', 0.5),
        ('scissors', 'paper', 1), ('scissors', 'rock', 0), ('scissors', 'scissors', 0.5)]

def load_or_create_dataset(load_dataset=True):
    """Memuat dataset jika ada, jika tidak buat dataset baru dengan augmentasi."""
    if os.path.exists(dataset_path) and load_dataset:
        print("Memuat dataset dari file...")
        df = pd.read_csv(dataset_path)
        X_train = df[['feature1', 'feature2']].values
        y_train = df['label'].values
    else:
        print("Membuat dataset baru dengan augmentasi...")
        le = LabelEncoder()
        le.fit(choices)
        
        X_train = np.array([[le.transform([p1])[0], le.transform([p2])[0]] for p1, p2, _ in data])
        X_train = X_train.astype(np.float32) / len(choices)  # Normalisasi
        
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)

        y_train = np.array([result for _, _, result in data])

        X_train, y_train = augment_data(X_train, y_train, noise_level=0.02, total_target=500)

        # Simpan ke CSV
        X_augmented_df = pd.DataFrame(X_train, columns=['feature1', 'feature2'])
        y_augmented_df = pd.DataFrame(y_train, columns=['label'])
        pd.concat([X_augmented_df, y_augmented_df], axis=1).to_csv(dataset_path, index=False)

        print(f"Dataset augmented disimpan di {dataset_path}")

    return X_train, y_train

def augment_data(X, y, noise_level=0.05, total_target=200):
    """Melakukan augmentasi data hingga jumlah total mencapai total_target."""
    
    n_original = len(X)
    if n_original >= total_target:
        print(f"Dataset sudah memiliki {n_original} baris, tidak perlu augmentasi tambahan.")
        return X, y  # Tidak perlu augmentasi jika sudah cukup

    n_needed = total_target - n_original
    num_augmented = max(1, n_needed // n_original)  # Pastikan minimal 1 augmentasi

    X_augmented = []
    y_augmented = []

    for _ in range(num_augmented):
        noise = np.random.normal(0, noise_level, X.shape)
        X_noisy = X + noise
        X_augmented.append(X_noisy)
        y_augmented.append(y)

    X_augmented = np.vstack(X_augmented)
    y_augmented = np.concatenate(y_augmented)

    # Gabungkan data asli dan augmentasi
    X_final = np.vstack((X, X_augmented))[:total_target]
    y_final = np.concatenate((y, y_augmented))[:total_target]

    print(f"Dataset setelah augmentasi: {len(X_final)} baris")
    return X_final, y_final

if __name__ == "__main__":
    X_train, y_train = load_or_create_dataset(load_dataset=False)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print("Contoh data:")
    print(X_train[:5])
    print(y_train[:5])
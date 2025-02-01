# model.py
# Model.py berisi kode untuk melatih model dan mengevaluasi model yang telah dilatih. 
# Kode ini akan melatih model Multi-Layer Perceptron (MLP) dengan teknik regulasi, 
# menyimpan model yang telah dilatih, dan menampilkan grafik loss dan accuracy dari model. 
# Kode ini juga akan mengevaluasi model dengan data validasi.

import os
# Konfigurasi TensorFlow agar tidak menampilkan log yang mengganggu
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

try:
    from src.data.dataset import load_or_create_dataset
except ModuleNotFoundError:
    from data.dataset import load_or_create_dataset

# mendapatkan path file ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Konstanta
MODEL_PATH = os.path.join(BASE_DIR, "mlp_model.keras")

# Path untuk menyimpan model dan history
BASE_HISTORY_PATH = os.path.join(BASE_DIR, "..", "artifacts")
SUMMARY_PATH = "model_summary.txt"
HISTORY_PATH = "history.csv"

def create_mlp_model():
    """Membangun model MLP dengan teknik regulasi."""
    model = keras.Sequential([
        keras.Input(shape=(2,)),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        keras.layers.Dropout(0.4),  # Dropout lebih tinggi
        keras.layers.BatchNormalization(), # Tambahkan Batch Normalization
        keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        keras.layers.Dropout(0.3),  # Tambahkan dropout lagi
        keras.layers.BatchNormalization(), # Tambahkan Batch Normalization
        keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        keras.layers.Dropout(0.2),  # Dropout lebih kecil di layer akhir
        keras.layers.BatchNormalization(), # Tambahkan Batch Normalization
        keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),  
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model


def train_or_load_model(load_model=True):
    """Memuat model jika ada, atau melatih model baru jika belum tersedia."""
    X_train, y_train = load_or_create_dataset(load_dataset=True)
    y_train = to_categorical(y_train, num_classes=3)
    
    if load_model and os.path.exists(MODEL_PATH):
        print("Memuat model yang sudah ada...")
        return keras.models.load_model(MODEL_PATH)

    print("Melatih model baru...")
    model = create_mlp_model()
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, 
                        epochs=200, 
                        verbose=1,  # Gunakan verbose=1 untuk melihat progres training
                        validation_split=0.3,  # 30% data untuk validasi
                        callbacks=[early_stopping, lr_scheduler
                                   ])

    # Simpan model dan history
    model.save(MODEL_PATH)
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(HISTORY_PATH, index=False)
    print(f"Model disimpan di {MODEL_PATH}")

    return model

def plot_loss_accuracy():
    """Menampilkan grafik loss & accuracy dalam satu plot."""
    if not os.path.exists(HISTORY_PATH):
        print("File history.csv tidak ditemukan.")
        return

    history = pd.read_csv(HISTORY_PATH)

    plt.figure(figsize=(10, 6))

    # Plot Loss
    plt.plot(history['loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='blue', linestyle='dashed')

    # Plot Accuracy
    plt.plot(history['accuracy'], label='Training Accuracy', color='green')
    plt.plot(history['val_accuracy'], label='Validation Accuracy', color='green', linestyle='dashed')

    last_value = history.iloc[-1]
    title = f"Training & Validation Loss and Accuracy\n accuracy: {last_value['accuracy']:.4f}, val_accuracy: {last_value['val_accuracy']:.4f}"
    print(title)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("history_plot.png")
    plt.show()

def evaluate_model(model):
    """Evaluasi model dengan data validasi."""
    X_train, y_train = load_or_create_dataset()
    y_train = to_categorical(y_train, num_classes=3)

    loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
    print(f"\nEvaluasi Model:\nLoss: {loss:.4f} - Akurasi: {accuracy:.4f}")

if __name__ == "__main__":
    model = train_or_load_model(load_model=False)
    plot_loss_accuracy()
    evaluate_model(model)

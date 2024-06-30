import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Data Preparation
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    X_train_normal = X_train[y_train == 0]
    X_test_normal = X_test[y_test == 0]
    X_test_fraud = X_test[y_test == 1]
    
    return X_train_normal, X_test_normal, X_test_fraud, X_test, y_test, scaler

# Model Building
def build_autoencoder(input_dim):
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    
    # Encoder
    encoder = tf.keras.layers.Dense(16, activation="relu")(input_layer)
    encoder = tf.keras.layers.Dense(8, activation="relu")(encoder)
    encoder = tf.keras.layers.Dense(4, activation="relu")(encoder)
    
    # Decoder
    decoder = tf.keras.layers.Dense(8, activation="relu")(encoder)
    decoder = tf.keras.layers.Dense(16, activation="relu")(decoder)
    decoder = tf.keras.layers.Dense(input_dim, activation="linear")(decoder)
    
    autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    return autoencoder

# Model Training
def train_model(model, X_train, X_test, epochs=50, batch_size=32):
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_test, X_test),
        verbose=1
    )
    return history

# Plotting
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_error_distribution(normal_error, fraud_error):
    plt.figure(figsize=(12, 6))
    plt.hist(normal_error, bins=50, alpha=0.5, label='Normal')
    plt.hist(fraud_error, bins=50, alpha=0.5, label='Fraud')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Distribution of Reconstruction Errors')
    plt.show()

# Anomaly Detection
def reconstruction_error(x, x_pred):
    return np.mean(np.square(x - x_pred), axis=1)

def detect_anomalies(model, X_test, X_test_normal, X_test_fraud, threshold_percentile=95):
    X_test_pred = model.predict(X_test)
    normal_error = reconstruction_error(X_test_normal, model.predict(X_test_normal))
    fraud_error = reconstruction_error(X_test_fraud, model.predict(X_test_fraud))
    
    threshold = np.percentile(normal_error, threshold_percentile)
    y_pred = (reconstruction_error(X_test, X_test_pred) > threshold).astype(int)
    
    return y_pred, normal_error, fraud_error, threshold

# Evaluation
def evaluate_model(y_true, y_pred):
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

# Single Transaction Anomaly Detection
def detect_anomaly(transaction, model, scaler, threshold):
    transaction = scaler.transform(transaction.reshape(1, -1))
    reconstruction = model.predict(transaction)
    error = reconstruction_error(transaction, reconstruction)
    is_anomaly = error > threshold
    return is_anomaly[0], error[0]

# Main execution
def main():
    # Load and prepare data
    X_train_normal, X_test_normal, X_test_fraud, X_test, y_test, scaler = load_and_prepare_data('creditcard.csv')
    
    # Build and train model
    input_dim = X_train_normal.shape[1]
    model = build_autoencoder(input_dim)
    history = train_model(model, X_train_normal, X_test_normal)
    
    # Plot training history
    plot_training_history(history)
    
    # Detect anomalies
    y_pred, normal_error, fraud_error, threshold = detect_anomalies(model, X_test, X_test_normal, X_test_fraud)
    
    # Plot error distribution
    plot_error_distribution(normal_error, fraud_error)
    
    # Evaluate model
    evaluate_model(y_test, y_pred)
    
    # Example of single transaction anomaly detection
    new_transaction = X_test[0]  # Using first transaction from test set as an example
    is_anomaly, error = detect_anomaly(new_transaction, model, scaler, threshold)
    print(f"\nSingle Transaction Analysis:")
    print(f"Is this transaction anomalous? {'Yes' if is_anomaly else 'No'}")
    print(f"Reconstruction error: {error}")
    print(f"Threshold: {threshold}")
    
    # Save the model
    model.save('credit_card_autoencoder.h5')

if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from keras_tuner import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters

# Data Preparation and Feature Engineering
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    df['Amount_Log'] = np.log(df['Amount'] + 1)
    df['Time_Bin'] = pd.cut(df['Time'], bins=24, labels=False)
    
    X = df.drop(['Class', 'Time', 'Amount'], axis=1)
    y = df['Class']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler

# Variational Autoencoder Model
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(hp):
    input_dim = X_train.shape[1]  # This needs to be passed as an argument
    
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    
    x = tf.keras.layers.Dense(hp.Int('dense_1', 32, 128, step=32), activation='relu')(inputs)
    x = tf.keras.layers.Dense(hp.Int('dense_2', 16, 64, step=16), activation='relu')(x)
    
    latent_dim = hp.Int('latent_dim', 2, 20, step=2)
    z_mean = tf.keras.layers.Dense(latent_dim)(x)
    z_log_var = tf.keras.layers.Dense(latent_dim)(x)
    z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])
    
    x = tf.keras.layers.Dense(hp.Int('dense_2', 16, 64, step=16), activation='relu')(z)
    x = tf.keras.layers.Dense(hp.Int('dense_1', 32, 128, step=32), activation='relu')(x)
    outputs = tf.keras.layers.Dense(input_dim, activation='linear')(x)
    
    vae = tf.keras.Model(inputs, outputs)
    
    reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
    reconstruction_loss *= input_dim
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_mean(kl_loss)
    kl_loss *= -0.5
    vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    
    vae.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])))
    
    return vae

# Hyperparameter Tuning
def tune_model(X_train, y_train):
    tuner = RandomSearch(
        build_vae,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=1,
        directory='vae_tuning',
        project_name='credit_card_anomaly'
    )
    
    class_weight = {0: 1, 1: (y_train == 0).sum() / (y_train == 1).sum()}
    
    tuner.search(X_train, X_train, 
                 epochs=10, 
                 validation_split=0.2, 
                 class_weight=class_weight)
    
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model

# Model Training
def train_model(model, X_train, X_test, y_train, epochs=50, batch_size=32):
    class_weight = {0: 1, 1: (y_train == 0).sum() / (y_train == 1).sum()}
    
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_test, X_test),
        class_weight=class_weight,
        verbose=1
    )
    return history

# Plotting
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    if 'loss' in history.history:
        plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
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

def detect_anomalies(model, X_test, y_test, threshold_percentile=95):
    X_test_pred = model.predict(X_test)
    errors = reconstruction_error(X_test, X_test_pred)
    
    normal_error = errors[y_test == 0]
    fraud_error = errors[y_test == 1]
    
    threshold = np.percentile(normal_error, threshold_percentile)
    y_pred = (errors > threshold).astype(int)
    
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
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data('creditcard.csv')
    
    best_model = tune_model(X_train, y_train)
    
    history = train_model(best_model, X_train, X_test, y_train)
    
    plot_training_history(history)
    
    y_pred, normal_error, fraud_error, threshold = detect_anomalies(best_model, X_test, y_test)
    
    plot_error_distribution(normal_error, fraud_error)
    
    evaluate_model(y_test, y_pred)
    
    new_transaction = X_test[0]
    is_anomaly, error = detect_anomaly(new_transaction, best_model, scaler, threshold)
    print(f"\nSingle Transaction Analysis:")
    print(f"Is this transaction anomalous? {'Yes' if is_anomaly else 'No'}")
    print(f"Reconstruction error: {error}")
    print(f"Threshold: {threshold}")
    
    best_model.save('credit_card_vae.h5')

if __name__ == "__main__":
    main()

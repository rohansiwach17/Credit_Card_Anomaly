# Credit Card Anomaly Detection

This project uses a Variational Autoencoder (VAE) to detect anomalies in credit card transactions, potentially identifying fraudulent activity. It incorporates advanced techniques such as feature engineering, hyperparameter tuning, and class imbalance handling.

## Features

- Variational Autoencoder (VAE) for robust anomaly detection
- Feature engineering to enhance model performance
- Hyperparameter tuning using Keras Tuner
- Class imbalance handling with weighted classes
- Visualization of training history and error distribution

## Setup

1. Clone this repository
2. Create a virtual environment: `python3 -m venv creditcard_anomaly_env`
3. Activate the virtual environment: `source creditcard_anomaly_env/bin/activate`
4. Install required packages: `pip install -r requirements.txt`
5. Run the script: `python credit_card_anomaly_detection.py`

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras Tuner
- Pandas
- Numpy
- Scikit-learn
- Matplotlib

## Usage

The main script `credit_card_anomaly_detection.py` performs the following steps:
1. Load and preprocess the credit card transaction data
2. Perform feature engineering
3. Tune hyperparameters and build a Variational Autoencoder
4. Train the model
5. Detect anomalies
6. Evaluate the model's performance
7. Demonstrate anomaly detection on a single transaction

## Results

The script outputs:
- Training history plot
- Distribution of reconstruction errors for normal and fraudulent transactions
- Confusion matrix and classification report
- Example of anomaly detection on a single transaction

## Future Improvements

- Experiment with different architectures (e.g., LSTM-VAE for temporal data)
- Implement more advanced feature engineering techniques
- Explore ensemble methods for improved performance
- Develop a real-time anomaly detection system

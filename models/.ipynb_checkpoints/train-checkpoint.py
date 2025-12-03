import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.lite.python.util import convert_bytes_to_string_or_bytearray
from pathlib import Path
import os

# --- 1. Generate Mock Data simulating the 13-column structure (A to M) ---

# In a real project, you would replace this section with your actual data loading:
df = pd.read_csv('sensor_data_2.csv', header=None)

# Mock Data Generation: 1000 rows, 13 columns (A-M)
# Column A (Index 0): Target (0-9)
# Columns B-M (Indices 1-12): 11 Sensor Features
num_samples = df.shape[0]
num_features = 12

# Create a DataFrame for clarity
df.columns = ['A_Target'] + [f'Feature_{i}' for i in range(1, 12)]

print(df.head())
print(f"Total samples: {df.shape[0]}, Total columns: {df.shape[1]}")

# --- 2. Data Preprocessing ---

# Separate features (X) and target (Y)
# X are columns 2 through 12 (C to M)
X = df.iloc[:, 2:].values
# Y is column 0 (A)
y = df.iloc[:, 0].values

# Standardize features (highly recommended for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert target labels to one-hot encoding (needed for multi-class classification)
y_encoded = keras.utils.to_categorical(y, num_classes=10) # 10 classes: 0 through 9

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# --- 3. Build and Train the Model ---

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(num_features,), name='input_layer'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    # Output layer for 10 classes (0-9)
    keras.layers.Dense(10, activation='softmax', name='output_layer')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- Model Training Started ---")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)
print("Model Training Complete.")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

# --- 4. Convert to TensorFlow Lite (.tflite) ---

# TFLite Converter requires a saved model
tf.saved_model.save(model, 'har_model_saved')

# Initialize the converter
converter = tf.lite.TFLiteConverter.from_saved_model('har_model_saved')

# OPTIMIZATION: Quantization for smaller size and faster inference on mobile
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # Enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # Enable select TensorFlow ops.
]

# Perform the conversion
tflite_model = converter.convert()

# Save the TFLite model to a file
tflite_model_path = Path('har_model.tflite')
tflite_model_path.write_bytes(tflite_model)
print(f"\nSuccessfully converted model to TFLite format and saved to: {tflite_model_path.resolve()}")

# --- 5. Prepare the Model Data for the Web App Simulation ---
# We convert the binary TFLite model to a base64 string for easy embedment 
# into the single-file HTML/JS app.

model_data_base64 = convert_bytes_to_string_or_bytearray(tflite_model)
print("\n--- TFLite Model Data (Base64) Generated for Deployment Simulation ---")
print("Copy the base64 string below into the 'tfliteModelBase64' variable in the HTML file.")

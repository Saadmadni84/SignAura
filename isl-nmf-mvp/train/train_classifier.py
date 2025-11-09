#!/usr/bin/env python3
"""
ISL NMF Classifier Training Script (Phase B)

This script trains a simple classifier to map NMF features to interpretations.
You can use this to replace the rule-based mapping in app.js.

Requirements:
  pip install pandas scikit-learn tensorflow numpy

Usage:
  python train_classifier.py

Output:
  - Trained model saved as 'nmf_classifier.pkl' (sklearn)
  - Or 'nmf_model/' (TensorFlow for web deployment)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import json

# Optional: TensorFlow for web deployment
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("TensorFlow not installed. Skipping TF.js export.")

# ============================================================================
# Configuration
# ============================================================================

DATASET_PATH = '../dataset/nmf_training.csv'
MODEL_TYPE = 'random_forest'  # 'random_forest' or 'logistic'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# Load Data
# ============================================================================

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"Loaded {len(df)} samples")
print(f"\nDataset preview:\n{df.head()}")
print(f"\nLabel distribution:\n{df['label'].value_counts()}")

# ============================================================================
# Prepare Features and Labels
# ============================================================================

feature_cols = [
    'eyebrow_raise',
    'mouth_open', 
    'head_tilt',
    'head_nod',
    'head_shake',
    'shoulder_lean'
]

X = df[feature_cols].values
y = df['label'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nFeatures shape: {X.shape}")
print(f"Labels shape: {y_encoded.shape}")
print(f"Number of classes: {len(label_encoder.classes_)}")

# ============================================================================
# Train-Test Split
# ============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print(f"\nTrain samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ============================================================================
# Train Classifier
# ============================================================================

print(f"\nTraining {MODEL_TYPE} classifier...")

if MODEL_TYPE == 'random_forest':
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE
    )
elif MODEL_TYPE == 'logistic':
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

model.fit(X_train, y_train)

# ============================================================================
# Evaluate
# ============================================================================

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"ACCURACY: {accuracy:.2%}")
print(f"{'='*60}")
print("\nClassification Report:")
print(classification_report(
    y_test, 
    y_pred,
    target_names=label_encoder.classes_
))

# ============================================================================
# Save Model (sklearn)
# ============================================================================

model_data = {
    'model': model,
    'label_encoder': label_encoder,
    'feature_names': feature_cols
}

with open('nmf_classifier.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\n✓ Model saved as 'nmf_classifier.pkl'")

# Save label mapping for JavaScript
label_mapping = {
    int(i): label 
    for i, label in enumerate(label_encoder.classes_)
}

with open('label_mapping.json', 'w') as f:
    json.dump(label_mapping, f, indent=2)

print("✓ Label mapping saved as 'label_mapping.json'")

# ============================================================================
# Optional: Train TensorFlow Model for Web Deployment
# ============================================================================

if HAS_TF:
    print("\nTraining TensorFlow model for web deployment...")
    
    # Build simple neural network
    tf_model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    tf_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    history = tf_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=4,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluate
    _, tf_accuracy = tf_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTensorFlow Model Accuracy: {tf_accuracy:.2%}")
    
    # Save in TensorFlow.js format
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(tf_model, 'nmf_model_tfjs')
    print("✓ TensorFlow.js model saved in 'nmf_model_tfjs/'")
    
    print("\nTo use in browser:")
    print("1. Copy 'nmf_model_tfjs/' to your web folder")
    print("2. Load with: tf.loadLayersModel('nmf_model_tfjs/model.json')")

# ============================================================================
# Feature Importance (for Random Forest)
# ============================================================================

if MODEL_TYPE == 'random_forest':
    print("\nFeature Importance:")
    for name, importance in zip(feature_cols, model.feature_importances_):
        print(f"  {name:20s}: {importance:.4f}")

# ============================================================================
# Example Predictions
# ============================================================================

print("\n" + "="*60)
print("Example Predictions:")
print("="*60)

examples = [
    ([1, 0, 0, 0, 0, 0], "Eyebrow raise only"),
    ([1, 1, 0, 0, 0, 0], "Eyebrow + mouth open"),
    ([0, 0, 0, 1, 0, 0], "Head nod"),
    ([0, 0, 0, 0, 1, 0], "Head shake"),
    ([1, 0, 0, 1, 0, 0], "Eyebrow + nod"),
    ([0, 0, 0, 0, 0, 0], "Neutral")
]

for features, description in examples:
    pred_idx = model.predict([features])[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    print(f"{description:30s} → {pred_label}")

print("\n✓ Training complete!")
print("\nNext steps:")
print("1. Collect more real data from webcam")
print("2. Re-run this script with larger dataset")
print("3. Integrate trained model into web app")
print("4. Optional: Export to TensorFlow.js for browser inference")

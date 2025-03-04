import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from utils import prepare_data

# Prepare the dataset
dataset_path = 'dataset'  # Update this path if needed
X, y = prepare_data(dataset_path)

# Encode labels: healthy -> 0, unhealthy -> 1
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Define the model
model = Sequential([
    Dense(128, input_dim=13, activation='relu'),  # Input layer size: 13 (MFCC features)
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary classification: healthy(0) or unhealthy(1)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y_encoded, epochs=20, batch_size=8, validation_split=0.2)

# Save the trained model
model.save('animal_health_model.h5')

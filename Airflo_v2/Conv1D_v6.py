import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("testpage_v5.csv")
labels = df[['Class_Temp']]

# Function to split the space-separated string into a list of integers
def split_binary_string(binary_string):
    return [int(x) for x in binary_string.split()]

# Apply the function to each column
day_features = np.array([split_binary_string(day) for day in df['Day']])
time_slot_features = np.array([split_binary_string(ts) for ts in df['Time_Slot']])
season_features = np.array([split_binary_string(season) for season in df['Season']])
week_features = np.array([split_binary_string(week) for week in df['Week']])
t_s_features = np.array(df['Total_Students']).reshape(-1, 1)
capac_features = np.array(df['Capacity']).reshape(-1, 1)

features = np.hstack((day_features, time_slot_features, season_features, week_features, t_s_features, capac_features))
features = features.astype(np.float32)
labels = labels.astype(np.float32)

# Normalize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Reshape features to add a channel dimension for Conv1D
features = features.reshape((features.shape[0], features.shape[1], 1))

# Split the data
X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.2, random_state=42)

# Model parameters
epochs = 900
batch_size = 256  # Reduce the batch size
lr = 0.0001  # Try a lower learning rate
adam = Adam(learning_rate=lr)

# Define the model
model_cnn = Sequential([
    Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(features.shape[1], 1)),
    BatchNormalization(),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.3), 
    Dense(1)
])

# Compile the model
model_cnn.compile(loss='mean_squared_error', optimizer=adam)
model_cnn.summary()

# Early stopping
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, 
                        verbose=1, mode='auto', restore_best_weights=True)

# Train the model
cnn_history = model_cnn.fit(X_train, y_train, callbacks=[monitor],
                            validation_data=(X_valid, y_valid), epochs=epochs, 
                            batch_size=batch_size, verbose=2)

# Predict on training and validation sets
cnn_train_pred = model_cnn.predict(X_train)
cnn_valid_pred = model_cnn.predict(X_valid)

# Evaluate the model
train_rmse = np.sqrt(mean_squared_error(y_train, cnn_train_pred))
valid_rmse = np.sqrt(mean_squared_error(y_valid, cnn_valid_pred))

print(f'Train RMSE: {train_rmse}')
print(f'Validation RMSE: {valid_rmse}')

# Plot training and validation loss
fig = plt.figure()
plt.plot(cnn_history.history['loss'], label='Train loss')
plt.plot(cnn_history.history['val_loss'], label='Validation loss')
fig.legend()
fig.suptitle('CNN Training and Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()

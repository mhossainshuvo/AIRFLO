import pandas as pd
import io
import os
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout
from keras.optimizers import Adam

# df = pd.read_csv("testpage.csv")
df = pd.read_csv("testpage_v5.csv")
# df = pd.read_csv("new_data.csv")
# features = df[['Day', 'Time_Slot', 'Season', 'Week', 'Total_Students', 'Capacity']]
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

# Reshape features to add a channel dimension for Conv1D
features = features.reshape((features.shape[0], features.shape[1], 1))

# Split the data
X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.2, random_state=42)

# Model parameters
epochs = 900
batch_size = 512
lr = 0.0003
adam = Adam(learning_rate=lr)

model_cnn = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(38, 1)),
    Conv1D(filters=32, kernel_size=2, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
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

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming y_train and y_valid are your actual temperature values for training and validation datasets
# Convert tensors to numpy arrays if needed
y_train = y_train.values.squeeze()
y_valid = y_valid.values.squeeze()

# Prepare data for plotting
train_data = pd.DataFrame({
    'Temperature': y_train,
    'Type': 'Train'
})
valid_data = pd.DataFrame({
    'Temperature': y_valid,
    'Type': 'Validation'
})

# Combine all data
plot_data = pd.concat([train_data, valid_data])

# Plot using Seaborn
plt.figure(figsize=(10, 6))
sns.kdeplot(data=train_data, x='Temperature', label='Train', fill=True)
sns.kdeplot(data=valid_data, x='Temperature', label='Validation', fill=True)

plt.title('Density Plot of Training and Validation Temperatures')
plt.xlabel('Temperature')
plt.ylabel('Density')
plt.legend(title='Dataset Type')
plt.show()

model_cnn.save('Hvac_model_cnn_v3.keras')

import pandas as pd
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
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout
from keras.optimizers import Adam

import pandas as pd

from keras.models import load_model

model = load_model('Hvac_model_cnn.keras')


# Example input
df = pd.read_csv("Test_data.csv")



# Encode categorical data as done in training
weekday_to_num = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5}
season_to_num = {'Fall': 1, 'Spring': 2, 'Summer': 3}

df['Day'] = df['Day'].replace(weekday_to_num)
df['Season'] = df['Season'].replace(season_to_num)

# Assuming Class and Time_Slot need similar transformation
class_to_num = {'Class 20': 20}  # This needs to be defined based on your training dataset
time_slot_to_num = {'Time slot 2': 2}  # Define similarly

df['Class'] = df['Class'].replace(class_to_num)
df['Time_Slot'] = df['Time_Slot'].replace(time_slot_to_num)

# Remove columns not used by the model
df = df[['Total_Students', 'Season_Temp', 'Class_Temp']]  # adjust based on actual model input

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(4, 1)))  # Adjust kernel_size to 2
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

X = np.array(df).reshape(1, df.shape[1], 1)  # Reshape to (num_samples, num_features, 1)

predicted_class_temp = model.predict(X)
print('Predicted Class Temperature:', predicted_class_temp[0][0])

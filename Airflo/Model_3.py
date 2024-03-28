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


df = pd.read_csv("Simulator.csv")

# Extracting integers from the 'Class' column and converting the result to integer
df['Class'] = df['Class'].str.extract('(\d+)').astype(int)

# Extracting integers from the 'Time_Slot' column and converting the result to integer
df['Time_Slot'] = df['Time_Slot'].str.extract('(\d+)').astype(int)

# Extracting integers from the 'Week' column and converting the result to integer
df['Week'] = df['Week'].str.extract('(\d+)').astype(int)

start_date = datetime(2022, 1, 3)

# Function to calculate date from the week number
def week_to_date(week):
    return start_date + timedelta(weeks=week-1)

# Create a new 'date' column
df['Date'] = df['Week'].apply(week_to_date)

def student_count(Student_Name):
    student_num = Student_Name.split()
    if len(student_num)==2:
        name_num = int(student_num[1])
    else:
        name_num = None
    return (name_num)
df['Student_Name'] = df['Student_Name'].apply(student_count)

weekday_to_num = {'Monday': 1,'Tuesday': 2,'Wednesday': 3,'Thursday': 4,'Friday': 5}
season_to_num = {'Fall': 1,'Spring': 2,'Summer': 3}

df['Day'] = df['Day'].replace(weekday_to_num)
df['Season'] = df['Season'].replace(season_to_num)
print(df.head(20))

def series_to_supervised(data, window=1, lag=1, dropnan=True):
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(window, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    # Current timestep (t=0)
    cols.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]
    # Target timestep (t=lag)
    cols.append(data.shift(-lag))
    names += [('%s(t+%d)' % (col, lag)) for col in data.columns]
    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Assuming 'df' is initial DataFrame

# Step 1: Calculate Aggregated Metrics
aggregated_data = df.groupby(['Class', 'Time_Slot', 'Week', 'Date'], as_index=False).agg({
    'Total_Students': 'mean',
    'Class_Temp': 'mean',
    'Season_Temp': 'mean'
}).rename(columns={'Total_Students': 'Average_Students', 'Class_Temp': 'Average_Class_Temp', 'Season_Temp': 'Average_Season_Temp'})

# Step 2: Merge Aggregated Metrics back into the original DataFrame
df = df.sort_values('Date')  # Ensure the DataFrame is sorted by 'Date'
df_enriched = df.merge(aggregated_data, on=['Class', 'Time_Slot', 'Week', 'Date'], how='left', suffixes=('', '_mean'))

# Replace original metrics with their mean values
df_enriched['Total_Students'] = df_enriched['Average_Students']
df_enriched['Class_Temp'] = df_enriched['Average_Class_Temp']
df_enriched['Season_Temp'] = df_enriched['Average_Season_Temp']

# Select only the desired columns to maintain the specified structure and exclude 'Student_Name'
df_final = df_enriched[['Class', 'Time_Slot', 'Week', 'Total_Students', 'Season_Temp', 'Class_Temp', 'Date']]

# This final DataFrame df_final is now sorted, excludes 'Student_Name', and incorporates the mean values for specified metrics
print(df_final)


window = 29
future_span = 30

# Dropping 'Day' for feature generation since it's a temporal marker not used in the same way as other features
series = series_to_supervised(df_final.drop('Date', axis=1), window=window, lag=future_span)
# print(series.head(10))

# Adjustments for edge case removal
last_class = 'Class(t-%d)' % window
last_time_slot = 'Time_Slot(t-%d)' % window
series = series[(series['Time_Slot(t)'] == series[last_time_slot])]
series = series[(series['Class(t)'] == series[last_class])]

# Adjustments for column dropping
# Assume interested in 'Total_Students', 'Season_Temp', and 'Class_Temp' as features to predict
columns_to_drop = [('%s(t+%d)' % (col, future_span)) for col in ['Class', 'Time_Slot']]
for i in range(window, 0, -1):
    columns_to_drop += [('%s(t-%d)' % (col, i)) for col in ['Class', 'Time_Slot']]
series.drop(columns_to_drop, axis=1, inplace=True)
series.drop(['Class(t)', 'Time_Slot(t)'], axis=1, inplace=True)  # Adjust if there are specific columns to retain for the model


# Label
labels_col = 'Class_Temp(t+%d)' % future_span
labels = series[labels_col]
series = series.drop(labels_col, axis=1)

# Splitting the dataset into training and validation sets
X_train, X_valid, Y_train, Y_valid = train_test_split(series, labels.values, test_size=0.4, random_state=0)
print('Train set shape', X_train.shape)
print('Validation set shape', X_valid.shape)

# If using models that require a 3D input shape (e.g., LSTM in Keras), reshape data like this:
X_train_series = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_valid_series = X_valid.values.reshape((X_valid.shape[0], X_valid.shape[1], 1))
print('Train set shape', X_train_series.shape)
print('Validation set shape', X_valid_series.shape)

epochs = 900
batch = 256
lr = 0.0003
adam = Adam(lr)

model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=8, activation='relu', input_shape=(X_train_series.shape[1], X_train_series.shape[2])))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dropout(0.2))
model_cnn.add(Dense(1))
model_cnn.compile(loss='mse', optimizer=adam)
model_cnn.summary()



monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, 
        verbose=1, mode='auto', restore_best_weights=True)

cnn_history = model_cnn.fit(X_train_series, Y_train, callbacks=[monitor],
    validation_data=(X_valid_series, Y_valid), epochs=epochs, verbose=2)


cnn_train_pred = model_cnn.predict(X_train_series)
cnn_valid_pred = model_cnn.predict(X_valid_series)
print('Train rmse:', np.sqrt(mean_squared_error(Y_train, cnn_train_pred)))
print('Validation rmse:', np.sqrt(mean_squared_error(Y_valid, cnn_valid_pred)))



fig = plt.figure()
plt.plot(cnn_history.history['loss'], label='Train loss')
plt.plot(cnn_history.history['val_loss'], label='Validation loss')
fig.legend()
fig.suptitle('CNN')
plt.xlabel("Epochs")
plt.ylabel("MSE")

plt.show()

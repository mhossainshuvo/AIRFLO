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
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# from keras.optimizers import adam_v2
from scipy import sparse
import spektral
from spektral.layers import GCNConv
from keras.models import Model
from keras.layers import Input, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from spektral.data import Graph

# Load data
df = pd.read_csv("Simulator.csv")
# Your data preprocessing remains the same

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

# Assume df_final is your processed DataFrame and ready to be used as described in your code

# Prepare graph data, Spektral uses a Graph object where you must define nodes, edges, and optionally edge features
# Here we use a simple example where each sample is fully connected and features are node attributes
# Typically, you need an adjacency matrix (A) and a node feature matrix (X)

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


# Simulate simple graph data
num_nodes = df_final.shape[0]
X = df_final.drop(['Date'], axis=1).values  # Node features
A = sparse.lil_matrix((num_nodes, num_nodes))
# A = np.ones((num_nodes, num_nodes))  # Fully connected graph (example)
y = df_final['Class_Temp'].values  # Labels for regression

# Create train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
A_train, A_test = A, A  # In a real scenario, you would have different adjacency matrices

# Define model
inputs = Input(shape=(X_train.shape[1],), sparse=False)
graph_inputs = Input((num_nodes,), sparse=True)

gc1 = GCNConv(32, activation='relu')([inputs, graph_inputs])
drop1 = Dropout(0.5)(gc1)
gc2 = GCNConv(32, activation='relu')([drop1, graph_inputs])
drop2 = Dropout(0.5)(gc2)
output = Dense(1)(drop2)
#

#
model = Model(inputs=[inputs, graph_inputs], outputs=output)
optimizer = Adam(lr=0.01)
model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
model.summary()

# Train model
validation_data = ([X_test, A_test], y_test)
model.fit([X_train, A_train], y_train, epochs=200, batch_size=32, validation_data=validation_data, verbose=2)

# Predict and evaluate
preds = model.predict([X_test, A_test])
test_rmse = np.sqrt(mean_squared_error(y_test, preds))
print('Test RMSE:', test_rmse)


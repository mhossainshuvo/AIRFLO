import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from keras.models import load_model

# Assuming the DataFrame is already loaded if not loaded we load it again
# df_test = pd.read_csv("Test_data.csv")

# Create a sample test DataFrame with structured data
# We create a DataFrame with one row and 124 features, ready for model input
data = {
    **{f'Total_Students(t-{i})': np.random.randint(50, 150) for i in range(30)},
    **{f'Class_Temp(t-{i})': np.random.uniform(50, 75) for i in range(30)},
    **{f'Season_Temp(t-{i})': np.random.uniform(30, 90) for i in range(30)},
    **{f'Placeholder_{i}(t)': np.random.random() for i in range(1, 34)}  # Additional placeholders
}

print(data)

# Create the DataFrame
df_test = pd.DataFrame([data])

# Normally you would apply transformations here if your test data was loaded from a file
# For our case, we assume df_test is already prepared as it needs to be

# Load your model
model = load_model('Hvac_model_cnn.keras')

# As our df_test is already prepared with the correct number of features we proceed directly to prediction
try:
    # Assuming the model expects data shaped as (samples, time steps, features)
    X_test = np.array(df_test).reshape(1, df_test.shape[1], 1)
    predicted_class_temp = model.predict(X_test)
    print('Predicted Class Temperature:', predicted_class_temp[0][0])
except Exception as e:
    print('Error during prediction:', str(e))

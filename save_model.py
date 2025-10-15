import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib

print("Starting model and scaler training...")

# --- 1. Load and Prepare the Full Dataset ---
df = pd.read_csv("insurance.csv")
df.drop_duplicates(inplace=True)

# Preprocessing
df['is_female'] = df['sex'].map({"male": 0, "female": 1})
df['is_smoker'] = df['smoker'].map({"no": 0, "yes": 1})
df = pd.get_dummies(df, columns=['region'], drop_first=True)
df['bmi_categories'] = pd.cut(df['bmi'], bins=[0, 18.5, 24.9, 29.9, float('inf')],
                              labels=['underweight', 'normal', 'overweight', 'obese'])
df = pd.get_dummies(df, columns=['bmi_categories'], drop_first=True)
df = df.drop(['sex', 'smoker'], axis=1)
df = df.astype(float)

# --- 2. Select Final Features and Scale the Data ---
final_df = df[[
    'age', 'bmi', 'is_smoker', 'children', 'charges',
    'region_southeast', 'is_female', 'bmi_categories_obese'
]]

x = final_df.drop('charges', axis=1)
y = final_df['charges']

# Initialize and fit the scaler on the feature data
scaler = StandardScaler()
cols_to_scale = ['age', 'bmi', 'children']
x[cols_to_scale] = scaler.fit_transform(x[cols_to_scale])

# --- 3. Train the Final Gradient Boosting Model ---
final_model = GradientBoostingRegressor(random_state=42)
final_model.fit(x, y)

# --- 4. Save the Model and the Scaler to Files ---
joblib.dump(final_model, 'gradient_boosting_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler have been saved successfully as 'gradient_boosting_model.pkl' and 'scaler.pkl'")
### Step 2: Build the Streamlit Web App



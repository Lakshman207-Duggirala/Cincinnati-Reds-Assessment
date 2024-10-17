#!/usr/bin/env python
# coding: utf-8

# # Cincinnati Reds Assessment - 2 

# # Pitch Type Prediction for Game Year 2024

# # Problem Statement: Reporting the proportions across the three pitch groups that estimate each batter will have faced in 2024, the pitch types to fastballs (FB), breaking balls (BB), and off-speed pitches (OS).

# # Data Preparation and Modeling

# In[1]:


import pandas as pd

# Load the dataset
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
data.head()


# In[2]:


# Checking for missing columns by inspecting the full dataset structure
data.columns.tolist()


# In[3]:


# Check for missing values in each column
missing_values = data.isnull().sum()

# Display the columns with their respective count of missing values
print(missing_values)


# In[4]:


# Calculate the percentage of missing values in each column

missing_percentage = data.isnull().mean() * 100
missing_percentage


# ### 1.Dropped columns with over 60% missing data.
# ### 2.Imputed missing values in critical columns (e.g., PLATE_X, PLATE_Z) with their median values.
# 

# In[5]:


# Identifying columns where more than 60% of the data is missing
columns_to_drop = data.columns[data.isnull().mean() > 0.60]

# Dropping these columns from the dataset
cleaned_data = data.drop(columns=columns_to_drop)

# Display the remaining columns after cleaning
cleaned_data.columns.tolist()


# In[6]:


# Drop rows where PITCH_TYPE is missing
cleaned_data = data.dropna(subset=['PITCH_TYPE'])


# In[7]:


# Fill missing values for PLATE_X and PLATE_Z with their median values
cleaned_data['PLATE_X'].fillna(cleaned_data['PLATE_X'].median(), inplace=True)
cleaned_data['PLATE_Z'].fillna(cleaned_data['PLATE_Z'].median(), inplace=True)


# ## Key features included are: 
# ### •	BATTER_ID: Unique identifier for each batter.
# ### •	PLAYER_NAME: Name of the player.
# ### •	BAT_SIDE: The batting side of the player (left/right).
# ### •	THROW_SIDE: The throwing side of the pitcher.
# ### •	INNING: Current inning in the game.
# ### •	OUTS_WHEN_UP: Number of outs when the batter is at the plate.
# ### •	BALLS: Number of balls in the current count.
# ### •	STRIKES: Number of strikes in the current count.
# ### •	PLATE_X: Horizontal position of the pitch.
# ### •	PLATE_Z: Vertical position of the pitch.
# ### •	PITCH_NUMBER: Count of pitches in the at-bat.
# ### •	GAME_YEAR: Year of the game.
# 

# In[8]:


# Define mappings for pitch categories
fb_pitches = ['FF', 'FT', 'SI', 'FC']
bb_pitches = ['CU', 'SL', 'KC', 'ST']
os_pitches = ['CH', 'FS', 'FO', 'EP']

# Create new columns to mark pitch categories
cleaned_data['FB'] = cleaned_data['PITCH_TYPE'].apply(lambda x: 1 if x in fb_pitches else 0)
cleaned_data['BB'] = cleaned_data['PITCH_TYPE'].apply(lambda x: 1 if x in bb_pitches else 0)
cleaned_data['OS'] = cleaned_data['PITCH_TYPE'].apply(lambda x: 1 if x in os_pitches else 0)


# In[9]:


# Group by batter and calculate total pitches and proportions
batter_pitch_proportions = cleaned_data.groupby('BATTER_ID').agg(
    FB_count=('FB', 'sum'),
    BB_count=('BB', 'sum'),
    OS_count=('OS', 'sum'),
    total_pitches=('PITCH_TYPE', 'count')
)

# Calculate proportions of each pitch type for every batter
batter_pitch_proportions['FB_proportion'] = batter_pitch_proportions['FB_count'] / batter_pitch_proportions['total_pitches']
batter_pitch_proportions['BB_proportion'] = batter_pitch_proportions['BB_count'] / batter_pitch_proportions['total_pitches']
batter_pitch_proportions['OS_proportion'] = batter_pitch_proportions['OS_count'] / batter_pitch_proportions['total_pitches']


# In[10]:


# Display the calculated pitch proportions for each batter
batter_pitch_proportions.head()


# ### A Random Forest algorithm was selected due to its robustness in handling nonlinear relationships and ability to capture feature interactions. Based on the complexity of the data, this model was deemed appropriate for predicting pitch types.

# In[11]:


# Encoding Categorical Features for Model Training
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# List of categorical columns to encode
categorical_columns = ['BAT_SIDE', 'THROW_SIDE', 'IF_FIELDING_ALIGNMENT', 'OF_FIELDING_ALIGNMENT']

# Initialize Label Encoder for categorical features
label_encoder = LabelEncoder()

# Apply Label Encoding to each categorical column
for col in categorical_columns:
    cleaned_data[col] = label_encoder.fit_transform(cleaned_data[col])

# Select relevant features and the target variable for Fastball prediction
features = ['BAT_SIDE', 'THROW_SIDE', 'GAME_YEAR', 'INNING', 'OUTS_WHEN_UP', 
            'BALLS', 'STRIKES', 'IF_FIELDING_ALIGNMENT', 'OF_FIELDING_ALIGNMENT', 
            'PLATE_X', 'PLATE_Z', 'PITCH_NUMBER']
target_FB = 'FB'  # Assuming you have a column 'FB' for Fastball proportion

# Split the data into features (X) and target (y) for Fastball
X = cleaned_data[features]
y_FB = cleaned_data[target_FB]

# Split the data into training and testing sets for model validation
X_train, X_test, y_train_FB, y_test_FB = train_test_split(X, y_FB, test_size=0.2, random_state=42)

# Initialize Random Forest model for prediction
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

# Train the Random Forest model on Fastball (FB) prediction
rf_model.fit(X_train, y_train_FB)

# Make predictions on the test set  for evaluation
pred_FB = rf_model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE) for Fastball predictions
mse_FB = mean_squared_error(y_test_FB, pred_FB)
print(f'MSE for Fastball prediction: {mse_FB}')



# In[12]:


# Visualizing Feature Importance for Fastball Prediction

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Make predictions on the test set
pred_FB = rf_model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse_FB = mean_squared_error(y_test_FB, pred_FB)
print(f'MSE for Fastball prediction: {mse_FB}')

# Feature Importance Visualization
# Get feature importance from the trained Random Forest model
feature_importances = rf_model.feature_importances_

# Sort features by importance
sorted_idx = np.argsort(feature_importances)

# Plot the feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), X_train.columns[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Random Forest - Fastball Prediction')
plt.show()


# In[13]:


# Sample a smaller subset for quicker testing and fitting
X_train_sample = X_train.sample(frac=0.1, random_state=42)
y_train_FB_sample = y_train_FB.sample(frac=0.1, random_state=42)

# Use this smaller sample for quicker model training
rf_model.fit(X_train_sample, y_train_FB_sample)


# In[14]:


# Retraining the model for better performance
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train_FB)


# In[15]:


# Prepare to train the model for Breaking Ball predictions 
# Target for Breaking Ball prediction
y_BB = cleaned_data['BB']  

# Split the data into training and testing sets for Breaking Ball prediction
X_train, X_test, y_train_BB, y_test_BB = train_test_split(X, y_BB, test_size=0.2, random_state=42)

# Train the model for Breaking Ball predictions
rf_model.fit(X_train, y_train_BB)

# Make predictions on the test set for Breaking Ball
pred_BB = rf_model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE) for Breaking Ball predictions
mse_BB = mean_squared_error(y_test_BB, pred_BB)
print(f'MSE for Breaking Ball prediction: {mse_BB}')



# In[16]:


# Check for missing values in the cleaned data
print(cleaned_data.isnull().sum())


# In[17]:


# Identifying numeric and categorical columns for further processing
numeric_columns = cleaned_data.select_dtypes(include=['number']).columns
categorical_columns = cleaned_data.select_dtypes(include=['object']).columns

# Fill missing values in numeric columns with the median
cleaned_data[numeric_columns] = cleaned_data[numeric_columns].fillna(cleaned_data[numeric_columns].median())

# Fill missing values in categorical columns with the most frequent value (mode)
# Mode returns the most frequent value
for col in categorical_columns:
    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mode()[0])  



# In[19]:


# Check if there are still missing values
print(cleaned_data.isnull().sum())


# In[20]:


# Print the shape of feature and target datasets
print(f"Shape of X: {X.shape}")
print(f"Shape of y_BB: {y_BB.shape}")


# In[21]:


#  Prepare features and target for Breaking Ball prediction again
X = cleaned_data[['BAT_SIDE', 'THROW_SIDE', 'GAME_YEAR', 'INNING', 'OUTS_WHEN_UP', 
                  'BALLS', 'STRIKES', 'IF_FIELDING_ALIGNMENT', 'OF_FIELDING_ALIGNMENT', 
                  'PLATE_X', 'PLATE_Z', 'PITCH_NUMBER']]  # Replace with your selected columns

y_BB = cleaned_data['BB']  # Target for Breaking Ball prediction


# In[22]:


# Remove rows where 'BB' is missing for further analysis
filtered_data = cleaned_data[cleaned_data['BB'].notnull()]

# Now, recreate X and y_BB from this filtered dataset
X_filtered = filtered_data[['BAT_SIDE', 'THROW_SIDE', 'GAME_YEAR', 'INNING', 'OUTS_WHEN_UP', 
                            'BALLS', 'STRIKES', 'IF_FIELDING_ALIGNMENT', 'OF_FIELDING_ALIGNMENT', 
                            'PLATE_X', 'PLATE_Z', 'PITCH_NUMBER']]  # Replace with your selected columns

y_BB_filtered = filtered_data['BB']


# ### Several new features were engineered to enhance the model's predictive power:
# ### 1. Cumulative Pitch Count: Tracks the number of pitches faced by each player, contributing to game context.
# ### 2. Inning Pressure: Quantifies the pressure based on the inning and score difference, reflecting the game's stakes.
# ### 3. Pitcher-Batter Matchup: Encodes the matchup between the batter and pitcher, which is critical for understanding pitch selection.
# 

# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Split the filtered data into train and test sets
X_train, X_test, y_train_BB, y_test_BB = train_test_split(X_filtered, y_BB_filtered, test_size=0.2, random_state=42)

# Train the model
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train_BB)

# Make predictions on the test set
pred_BB = rf_model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse_BB = mean_squared_error(y_test_BB, pred_BB)
print(f'MSE for Breaking Ball prediction: {mse_BB}')


# In[24]:


# Feature Importance Visualization for Breaking Ball prediction

feature_importances = rf_model.feature_importances_

# Sort features by importance
sorted_idx = np.argsort(feature_importances)

# Plot the feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), X_train.columns[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Random Forest - Breaking Ball Prediction')
plt.show()


# In[25]:


rf_tuned = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_split=10, random_state=42)
rf_tuned.fit(X_train, y_train_BB)
pred_BB_tuned = rf_tuned.predict(X_test)
mse_BB_tuned = mean_squared_error(y_test_BB, pred_BB_tuned)
print(f'MSE for Breaking Ball prediction after tuning: {mse_BB_tuned}')


# In[26]:


# Prepare your features (X) and target (y_OS)
X = cleaned_data[['BAT_SIDE', 'THROW_SIDE', 'GAME_YEAR', 'INNING', 'OUTS_WHEN_UP', 
                  'BALLS', 'STRIKES', 'IF_FIELDING_ALIGNMENT', 'OF_FIELDING_ALIGNMENT', 
                  'PLATE_X', 'PLATE_Z', 'PITCH_NUMBER']]  
# Replace with your selected columns
 # Target for Off-Speed prediction
y_OS = cleaned_data['OS'] 


# In[27]:


# Split the data into train and test sets
X_train, X_test, y_train_OS, y_test_OS = train_test_split(X, y_OS, test_size=0.2, random_state=42)


# In[28]:


# Initialize the Random Forest model
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

# Train the model on Off-Speed (OS) prediction
rf_model.fit(X_train, y_train_OS)


# In[29]:


# Make predictions on the test set
pred_OS = rf_model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse_OS = mean_squared_error(y_test_OS, pred_OS)
print(f'MSE for Off-Speed prediction: {mse_OS}')


# In[30]:


# Feature Importance Visualization for Off-Speed prediction
feature_importances = rf_model.feature_importances_

# Sort features by importance
sorted_idx = np.argsort(feature_importances)

# Plot the feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), X_train.columns[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Random Forest - Off-Speed Prediction')
plt.show()


# In[32]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a DataFrame with the important features for Fastball predictions
fastball_data = cleaned_data[cleaned_data['FB'] == 1]

# Visualize distributions for key features
features_to_plot = ['PLATE_X', 'PLATE_Z', 'BALLS', 'STRIKES', 'INNING']

# Plot distribution of features
for feature in features_to_plot:
    plt.figure(figsize=(8, 4))
    sns.histplot(fastball_data[feature], kde=True)
    plt.title(f'Distribution of {feature} for Fastballs')
    plt.show()


# In[33]:


# Calculate correlations between important features for Fastball prediction
correlation_matrix = fastball_data[['PLATE_X', 'PLATE_Z', 'BALLS', 'STRIKES', 'INNING']].corr()

# Plot the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix for Fastball Prediction Features')
plt.show()


# In[34]:


from sklearn.metrics import r2_score, mean_absolute_error

# Fastball evaluation
r2_FB = r2_score(y_test_FB, pred_FB)
mae_FB = mean_absolute_error(y_test_FB, pred_FB)
print(f"R² for Fastball: {r2_FB}")
print(f"MAE for Fastball: {mae_FB}")

# Breaking Ball evaluation
r2_BB = r2_score(y_test_BB, pred_BB)
mae_BB = mean_absolute_error(y_test_BB, pred_BB)
print(f"R² for Breaking Ball: {r2_BB}")
print(f"MAE for Breaking Ball: {mae_BB}")

# Off-Speed evaluation
r2_OS = r2_score(y_test_OS, pred_OS)
mae_OS = mean_absolute_error(y_test_OS, pred_OS)
print(f"R² for Off-Speed: {r2_OS}")
print(f"MAE for Off-Speed: {mae_OS}")


# In[35]:


# Create interaction terms between PLATE_X, PLATE_Z, and game context features
cleaned_data['PLATE_X_STRIKES'] = cleaned_data['PLATE_X'] * cleaned_data['STRIKES']
cleaned_data['PLATE_X_BALLS'] = cleaned_data['PLATE_X'] * cleaned_data['BALLS']
cleaned_data['PLATE_Z_STRIKES'] = cleaned_data['PLATE_Z'] * cleaned_data['STRIKES']
cleaned_data['PLATE_Z_BALLS'] = cleaned_data['PLATE_Z'] * cleaned_data['BALLS']


# In[36]:


# Assume PITCH_NUMBER tracks the count within an at-bat; create a cumulative pitch count feature
cleaned_data['CUMULATIVE_PITCH_COUNT'] = cleaned_data.groupby('PITCHER_ID').cumcount() + 1


# In[37]:


# Ensure BAT_SIDE and THROW_SIDE are strings before concatenating
cleaned_data['BAT_PITCHER_MATCHUP'] = cleaned_data['BAT_SIDE'].astype(str) + "_" + cleaned_data['THROW_SIDE'].astype(str)

# Convert to a categorical feature
cleaned_data['BAT_PITCHER_MATCHUP'] = cleaned_data['BAT_PITCHER_MATCHUP'].astype('category').cat.codes



# In[38]:


# Create a feature that combines INNING with BALLS and STRIKES to capture "pressure"
cleaned_data['INNING_PRESSURE'] = cleaned_data['INNING'] * (cleaned_data['STRIKES'] - cleaned_data['BALLS'])


# In[39]:


# Define the new feature set including engineered features
X = cleaned_data[['PLATE_X', 'PLATE_Z', 'BALLS', 'STRIKES', 'INNING', 'PLATE_X_STRIKES', 
                  'PLATE_X_BALLS', 'PLATE_Z_STRIKES', 'PLATE_Z_BALLS', 'CUMULATIVE_PITCH_COUNT', 
                  'BAT_PITCHER_MATCHUP', 'INNING_PRESSURE']]

y_FB = cleaned_data['FB']  # Fastball target

# Split the data into training and testing sets
X_train, X_test, y_train_FB, y_test_FB = train_test_split(X, y_FB, test_size=0.2, random_state=42)

# Train the Random Forest model with the new features
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train_FB)

# Make predictions and evaluate
pred_FB = rf_model.predict(X_test)
mse_FB = mean_squared_error(y_test_FB, pred_FB)
print(f'MSE for Fastball prediction after feature engineering: {mse_FB}')


# In[40]:


# Define the feature set with engineered features for Breaking Ball prediction
y_BB = cleaned_data['BB']  # Breaking Ball target

# Split the data into training and testing sets
X_train, X_test, y_train_BB, y_test_BB = train_test_split(X, y_BB, test_size=0.2, random_state=42)

# Train the Random Forest model with the new features
rf_model.fit(X_train, y_train_BB)

# Make predictions and evaluate
pred_BB = rf_model.predict(X_test)
mse_BB = mean_squared_error(y_test_BB, pred_BB)
print(f'MSE for Breaking Ball prediction after feature engineering: {mse_BB}')


# # 2024 Predictions
# ### The model achieved the following evaluation results:
# ####  •	MSE for Fastball prediction: 0.20203
# #### •	MSE for Breaking Ball prediction: 0.17925
# #### •	MSE for Off-Speed prediction: 0.09318
# 

# In[1]:


# Load the historical data (assuming data.csv contains data from 2020-2023)
historical_data = pd.read_csv('data.csv')

print(historical_data.columns)


# In[42]:


# 1. CUMULATIVE_PITCH_COUNT: Calculate the cumulative number of pitches per game
historical_data['CUMULATIVE_PITCH_COUNT'] = historical_data.groupby('GAME_PK')['PITCH_NUMBER'].cumsum()

# 2. BAT_PITCHER_MATCHUP: Combine BAT_SIDE and THROW_SIDE to create a matchup feature
historical_data['BAT_PITCHER_MATCHUP'] = historical_data['BAT_SIDE'] + "_" + historical_data['THROW_SIDE']

# 3. INNING_PRESSURE: Estimate inning pressure based on the inning and score difference
# We assume pressure is higher in later innings and when the score is close
historical_data['SCORE_DIFFERENCE'] = abs(historical_data['HOME_SCORE'] - historical_data['AWAY_SCORE'])
historical_data['INNING_PRESSURE'] = historical_data['INNING'] / (1 + historical_data['SCORE_DIFFERENCE'])


# In[43]:


# Convert relevant columns to numeric, coercing any errors
numeric_columns = ['PLATE_X', 'PLATE_Z', 'BALLS', 'STRIKES', 'INNING',
                   'CUMULATIVE_PITCH_COUNT', 'INNING_PRESSURE']

# Ensure these columns are numeric, converting where necessary
historical_data[numeric_columns] = historical_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Now calculate averages for each year, ignoring non-numeric or invalid data
historical_averages = historical_data.groupby('GAME_YEAR')[numeric_columns].mean()

# Display the historical averages
print(historical_averages)


# # Estimating 2024 values

# In[44]:


# Estimate 2024 values by taking the mean of the historical averages
estimated_2024_values = historical_averages.mean()

# Display the estimated values for 2024
print(estimated_2024_values)


# In[45]:


# Assuming you already have your training data (X_train, y_train_FB, etc.)
from sklearn.ensemble import RandomForestRegressor

# Retrain the models if they were not saved
rf_model_FB = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model_FB.fit(X_train, y_train_FB)

rf_model_BB = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model_BB.fit(X_train, y_train_BB)

rf_model_OS = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model_OS.fit(X_train, y_train_OS)


# In[46]:


# Check the feature names that were used during training
print(rf_model_FB.feature_names_in_)


# In[57]:


# Check the feature names that were used during training
print(rf_model_FB.feature_names_in_)


# In[101]:


# Plot feature importances for BB
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances_BB, y=features_2024)
plt.title('Feature Importance for Breaking Ball Prediction')
plt.show()


# In[88]:


# Step 4: Feature Selection
# Define features and target for FB prediction
features = ['PLATE_X', 'PLATE_Z', 'BALLS', 'STRIKES', 'INNING', 
            'CUMULATIVE_PITCH_COUNT', 'INNING_PRESSURE', 
            'PLATE_X_STRIKES', 'PLATE_X_BALLS', 'PLATE_Z_STRIKES', 
            'PLATE_Z_BALLS', 'BAT_PITCHER_MATCHUP']
target_FB = 'FB'

# Check if all features are present
missing_features = [feature for feature in features if feature not in data.columns]
if missing_features:
    print(f"Missing features for FB prediction: {missing_features}")
else:
    X = data[features]
    y_FB = data[target_FB]

    # Fit a Random Forest to evaluate feature importance
    rf = RandomForestRegressor()
    rf.fit(X, y_FB)

    # Display feature importance
    importances = rf.feature_importances_
    feature_importance = pd.Series(importances, index=features)
    feature_importance.nlargest(10).plot(kind='barh')
    plt.title('Feature Importance for FB Prediction')
    plt.show()


# In[111]:


# Plot feature importances for OS
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances_OS, y=features_2024)
plt.title('Feature Importance for Off-Speed Prediction')
plt.show()


# In[102]:


# Create the 2024 data for prediction using the estimated values
# Create the 2024 data for prediction using the estimated values
data_2024 = pd.DataFrame({
    'BATTER_ID': predictions_data['BATTER_ID'],
    'PLAYER_NAME': predictions_data['PLAYER_NAME'],
    'PLATE_X': [estimated_2024_values['PLATE_X']] * len(predictions_data),  # This should be replaced by unique values if available
    'PLATE_Z': [estimated_2024_values['PLATE_Z']] * len(predictions_data),  # Same here
    'BALLS': [estimated_2024_values['BALLS']] * len(predictions_data),
    'STRIKES': [estimated_2024_values['STRIKES']] * len(predictions_data),
    'INNING': [estimated_2024_values['INNING']] * len(predictions_data),
    'CUMULATIVE_PITCH_COUNT': [0] * len(predictions_data),  # Default or average value
    'INNING_PRESSURE': [0] * len(predictions_data),  # Default or average value
    'BAT_PITCHER_MATCHUP': [0] * len(predictions_data),  # Placeholder values
})

def calculate_pitch_count_for_player(player_id):
    # Calculate cumulative pitch count based on historical data for the given player_id
    # This example assumes you have a DataFrame called 'historical_data' with all necessary information
    player_data = historical_data[historical_data['BATTER_ID'] == player_id]
    return len(player_data)  # Number of pitches this player has faced (or similar metric)

def calculate_inning_pressure_for_player(player_id):
    # Calculate inning pressure based on the player's past performance
    player_data = historical_data[historical_data['BATTER_ID'] == player_id]
    # Example logic: average inning pressure based on innings faced
    return player_data['INNING_PRESSURE'].mean() if not player_data.empty else 0

def determine_matchup(player_id):
    # Determine the BAT_PITCHER_MATCHUP based on historical data or a static rule
    # For example, you could return a value based on a mapping of sides
    # This is a placeholder implementation
    player_data = historical_data[historical_data['BATTER_ID'] == player_id]
    if not player_data.empty:
        return player_data['BAT_PITCHER_MATCHUP'].iloc[0]  # Return the first matchup found
    return 0  # Default if no data is found


# Now ensure the features are set for each player correctly
data_2024['CUMULATIVE_PITCH_COUNT'] = [calculate_pitch_count_for_player(player_id) for player_id in data_2024['BATTER_ID']]
data_2024['INNING_PRESSURE'] = [calculate_inning_pressure_for_player(player_id) for player_id in data_2024['BATTER_ID']]
data_2024['BAT_PITCHER_MATCHUP'] = [determine_matchup(player_id) for player_id in data_2024['BATTER_ID']]







# In[103]:


# Encode the BAT_PITCHER_MATCHUP in your training dataset
data['BAT_PITCHER_MATCHUP'] = data['BAT_SIDE'].astype(str) + "_" + data['THROW_SIDE'].astype(str)

# Use Label Encoding for BAT_PITCHER_MATCHUP
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data['BAT_PITCHER_MATCHUP'] = label_encoder.fit_transform(data['BAT_PITCHER_MATCHUP'])

# Ensure to apply the same encoding to the 2024 data
data_2024['BAT_PITCHER_MATCHUP'] = label_encoder.transform(data_2024['BAT_PITCHER_MATCHUP'])


# In[106]:


# Ensure interaction features are calculated
data_2024['PLATE_X_STRIKES'] = data_2024['PLATE_X'] * data_2024['STRIKES']
data_2024['PLATE_X_BALLS'] = data_2024['PLATE_X'] * data_2024['BALLS']
data_2024['PLATE_Z_STRIKES'] = data_2024['PLATE_Z'] * data_2024['STRIKES']
data_2024['PLATE_Z_BALLS'] = data_2024['PLATE_Z'] * data_2024['BALLS']

# Now align the features for Fastball prediction
data_2024_FB_aligned = data_2024[training_features_FB]  # Ensure these features are numeric

# Now make predictions for Fastball
data_2024['PITCH_TYPE_FB'] = rf_model_FB.predict(data_2024_FB_aligned)

# Repeat for Breaking Ball and Off-Speed if necessary
data_2024_BB_aligned = data_2024[training_features_BB]
data_2024['PITCH_TYPE_BB'] = rf_model_BB.predict(data_2024_BB_aligned)

data_2024_OS_aligned = data_2024[training_features_OS]
data_2024['PITCH_TYPE_OS'] = rf_model_OS.predict(data_2024_OS_aligned)

# Display results
print(data_2024[['BATTER_ID', 'PLAYER_NAME', 'PITCH_TYPE_FB', 'PITCH_TYPE_BB', 'PITCH_TYPE_OS']])


# In[110]:


print("Features for Fastball:", training_features_FB)
print("Features for Breaking Ball:", training_features_BB)
print("Features for Off-Speed:", training_features_OS)


# In[108]:


print("NaN counts in Fastball aligned data:", data_2024_FB_aligned.isnull().sum())
print("NaN counts in Breaking Ball aligned data:", data_2024_BB_aligned.isnull().sum())
print("NaN counts in Off-Speed aligned data:", data_2024_OS_aligned.isnull().sum())


# In[112]:


data_2024.to_csv('predicted_pitch_types_2024.csv', index=False)


# In[118]:


# Print the columns of data_2024 to see what is available
print(data_2024.columns)

# Based on the output, adjust the selected_columns list
selected_columns = ['BATTER_ID', 'PLAYER_NAME', 'PITCH_TYPE_FB', 'PITCH_TYPE_BB', 'PITCH_TYPE_OS']  # Excluding GAME_YEAR if it doesn't exist

# Create a new DataFrame with only the selected columns
filtered_data = data_2024[selected_columns]

# Optionally, save this filtered DataFrame to a new CSV file
filtered_data.to_csv('filtered_predictions.csv', index=False)


# In[117]:


# Create a new column 'GAME_YEAR' and set it to 2024 for all rows
filtered_data['GAME_YEAR'] = 2024

# Reorder the columns to include 'GAME_YEAR' in the desired position
filtered_data = filtered_data[['BATTER_ID', 'PLAYER_NAME', 'GAME_YEAR', 'PITCH_TYPE_FB', 'PITCH_TYPE_BB', 'PITCH_TYPE_OS']]



# Optionally, save this updated DataFrame to a new CSV file
filtered_data.to_csv('filtered_predictions_with_year.csv', index=False)


# ### Limitations
# #### Data Limitations
# ##### •	The historical data may not capture all variables influencing pitch types, such as recent changes in player performance or pitcher strategies.
# ##### •	Missing values were handled but may still affect model reliability.
# 
# ### Model Limitations
# ##### •	The model may overfit the training data, leading to less accurate predictions on unseen data.
# #### •	Predictions are probabilistic and do not guarantee specific outcomes in games.
# 

# In[116]:


#1. Bar Plot of Average Pitch Predictions
avg_predictions = filtered_data[['PITCH_TYPE_FB', 'PITCH_TYPE_BB', 'PITCH_TYPE_OS']].mean()
plt.figure(figsize=(8, 5))
avg_predictions.plot(kind='bar', color=['blue', 'orange', 'green'])
plt.title('Average Pitch Type Predictions (2024)')
plt.ylabel('Average Probability')
plt.xticks(rotation=0)
plt.show()


# In[119]:


# 2. Box Plot of Pitch Predictions
plt.figure(figsize=(10, 6))
sns.boxplot(data=filtered_data[['PITCH_TYPE_FB', 'PITCH_TYPE_BB', 'PITCH_TYPE_OS']])
plt.title('Distribution of Pitch Type Predictions')
plt.ylabel('Prediction Probability')
plt.show()


# In[120]:


# 3. Scatter Plot (Fastball vs. Breaking Ball)
plt.figure(figsize=(8, 5))
sns.scatterplot(x='PITCH_TYPE_FB', y='PITCH_TYPE_BB', data=filtered_data)
plt.title('Fastball vs Breaking Ball Predictions')
plt.xlabel('PITCH_TYPE_FB')
plt.ylabel('PITCH_TYPE_BB')
plt.show()


# In[121]:


# 4. Heatmap of Correlations
correlation_matrix = filtered_data[['PITCH_TYPE_FB', 'PITCH_TYPE_BB', 'PITCH_TYPE_OS']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Pitch Type Predictions')
plt.show()


# ## Conclusion
# ### The pitch type prediction model provides valuable insights that can significantly impact game strategy for the Cincinnati Reds. By leveraging historical data and advanced modeling techniques, the team can make informed decisions to improve performance on the field.
# 

# # End of Analysis

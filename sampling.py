import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Download the dataset

df = pd.read_csv("Creditcard_data.csv")

# Identify the target variable
target_variable = df.columns[-1]

# Balance the dataset using RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(df.drop(target_variable, axis=1), df[target_variable])

# Scale the features
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

# Split features and target variable
X = X_resampled_scaled
y = y_resampled

# Define TOPSIS function
def topsis_score(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return ((accuracy + precision + recall + f1) / 4) * 100

# Define ML models
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),  # Set probability=True for AUC-ROC
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

# Define sampling techniques
samplings = {
    'RandomUnderSampling': RandomUnderSampler(random_state=42),
    'RandomOverSampling': RandomOverSampler(random_state=42),
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42)
}

# Results list
results_list = []

# Apply different sampling techniques on different ML models
for model_name, model in models.items():
    for sampling_name, sampling in samplings.items():
        X_resampled, y_resampled = sampling.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        topsis = topsis_score(y_test, y_pred)

        results_list.append({
            'Model': model_name,
            'Sampling Technique': sampling_name,
            'TOPSIS Score': topsis
        })

# Create a DataFrame from the results list
results_df = pd.DataFrame(results_list)

# Pivot the DataFrame for tabulating into a table
table_df = results_df.pivot(index='Model', columns='Sampling Technique', values='TOPSIS Score')

# Save the DataFrame to a CSV file named "result.csv"
csv_file_path = "result.csv"
table_df.to_csv(csv_file_path)

print(f"Table with TOPSIS scores saved to {csv_file_path}")

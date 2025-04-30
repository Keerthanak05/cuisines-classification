import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
import matplotlib.pyplot as plt
import seaborn as sns
from uuid import uuid4

# Step 1: Load and Preprocess Dataset
df = pd.read_csv('Dataset .csv')
df.columns = df.columns.str.strip()  # Clean column names
print("📄 Columns in dataset:", df.columns.tolist())
print("\n🔍 Dataset preview:\n", df.head())

# Handle missing values in 'Cuisines'
df['Cuisines'] = df['Cuisines'].fillna('Unknown')  # Replace NaN with 'Unknown'
print("\n🧹 Missing values before preprocessing:\n", df.isnull().sum())

# Drop non-informative columns
drop_cols = ['Restaurant ID', 'Restaurant Name', 'Address', 'Locality', 'Locality Verbose', 'Longitude', 'Latitude']
df = df.drop(columns=drop_cols)

# Convert 'Cuisines' to list of cuisines for multi-label classification
df['Cuisines'] = df['Cuisines'].apply(lambda x: [c.strip() for c in x.split(',')])

# Encode categorical variables
categorical_cols = ['City', 'Currency', 'Has Table booking', 'Has Online delivery', 
                    'Is delivering now', 'Switch to order menu', 'Rating color', 'Rating text']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Handle multi-label target (Cuisines)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['Cuisines'])
print(f"✅ Number of cuisine labels: {len(mlb.classes_)}")
print("Cuisine labels:", mlb.classes_)

# Drop Cuisines from features
X = df.drop(columns=['Cuisines'])

# Handle any remaining missing values in numerical columns
numerical_cols = ['Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# Step 2: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Step 3: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train Model
base_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model = MultiOutputClassifier(base_rf, n_jobs=-1)
model.fit(X_train, y_train)
print("✅ Model training completed")

# Step 5: Evaluate Model
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
hamming = hamming_loss(y_test, y_pred)
print("\n📊 Evaluation Metrics:")
print(f"Exact Match Accuracy: {accuracy:.4f}")
print(f"Hamming Loss: {hamming:.4f}")

# Detailed classification report for each label
print("\n📈 Classification Report for Each Cuisine:")
report = classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0, output_dict=True)
for label, metrics in report.items():
    if label not in ['micro avg', 'macro avg', 'weighted avg', 'samples avg']:
        print(f"\nCuisine: {label}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1-score']:.4f}")
        print(f"Support: {metrics['support']}")

# Aggregate metrics
print("\n📊 Aggregate Metrics:")
print(f"Micro Avg Precision: {report['micro avg']['precision']:.4f}")
print(f"Micro Avg Recall: {report['micro avg']['recall']:.4f}")
print(f"Micro Avg F1-Score: {report['micro avg']['f1-score']:.4f}")
print(f"Macro Avg Precision: {report['macro avg']['precision']:.4f}")
print(f"Macro Avg Recall: {report['macro avg']['recall']:.4f}")
print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")

# Step 6: Analyze Performance
# Calculate label frequency in test set
label_freq = y_test.sum(axis=0)
freq_df = pd.DataFrame({'Cuisine': mlb.classes_, 'Frequency': label_freq})
freq_df = freq_df.sort_values(by='Frequency', ascending=False)

# Plot label frequency
plt.figure(figsize=(12, 6))
sns.barplot(x='Frequency', y='Cuisine', data=freq_df.head(20))
plt.title('Top 20 Cuisine Label Frequencies in Test Set')
plt.xlabel('Number of Occurrences')
plt.ylabel('Cuisine')
plt.tight_layout()
plt.show()

# Analyze feature importance (average across all classifiers)
feature_importance = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df.head(10))
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

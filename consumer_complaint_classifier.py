# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('/content/complaints.csv')

# Data Preprocessing: Drop rows with missing narratives
df = df.dropna(subset=['Consumer complaint narrative'])

# Map product categories to numeric labels
product_to_category = {
    'Credit reporting': 0,
    'Credit reporting, credit repair services, or other personal consumer reports': 0,
    'Debt collection': 1,
    'Mortgage': 3,
    'Payday loan': 2,
    'Payday loan, title loan, or personal loan': 2,
    'Student loan': 2,
    'Vehicle loan or lease': 2,
    'Personal loan': 2,
    'Installment loan': 2
}

# Filter dataset to only relevant categories
df = df[df['Product'].isin(product_to_category.keys())]
df['Category'] = df['Product'].map(product_to_category)

# Balance dataset by sampling up to 10,000 entries per category
df_sample = df.groupby('Category').apply(lambda x: x.sample(min(10000, len(x)), random_state=42)).reset_index(drop=True)

# Exploratory Data Analysis (EDA)
print("=== Exploratory Data Analysis ===")
print("Dataset Shape:", df_sample.shape)
print("\nCategory Distribution:")
print(df_sample['Category'].value_counts())

# Analyze narrative length
df_sample['Narrative_Length'] = df_sample['Consumer complaint narrative'].apply(len)
print("\nAverage Narrative Length per Category:")
print(df_sample.groupby('Category')['Narrative_Length'].mean())

# Visualize category distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Category', data=df_sample)
plt.title('Distribution of Complaint Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

# Prepare features (X) and labels (y)
X = df_sample['Consumer complaint narrative']
y = df_sample['Category']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("\nTF-IDF Feature Matrix Shape (Train):", X_train_tfidf.shape)
print("TF-IDF Feature Matrix Shape (Test):", X_test_tfidf.shape)

# Initialize models
models = {
    'Multinomial Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs'),
    'Linear SVC': LinearSVC(max_iter=1000)
}

# Model Training and Evaluation
print("\n=== Model Performance Comparison ===")
best_model = None
best_accuracy = 0
y_pred_best = None

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{name} Accuracy: {accuracy:.4f}")
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred, target_names=['Credit Reporting', 'Debt Collection', 'Consumer Loan', 'Mortgage']))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = name
        y_pred_best = y_pred

print(f"\nBest Model: {best_model} with Accuracy: {best_accuracy:.4f}")

# Confusion Matrix Visualization
print("\n=== Model Evaluation ===")
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Credit Reporting', 'Debt Collection', 'Consumer Loan', 'Mortgage'],
            yticklabels=['Credit Reporting', 'Debt Collection', 'Consumer Loan', 'Mortgage'])
plt.title(f'Confusion Matrix for {best_model}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Predicting New Complaint
new_complaint = ["I have an issue with my credit report being inaccurate."]
new_complaint_tfidf = vectorizer.transform(new_complaint)

svc = LinearSVC(max_iter=1000)
svc.fit(X_train_tfidf, y_train)
predicted_category = svc.predict(new_complaint_tfidf)

category_names = {0: 'Credit Reporting, Repair, or Other', 1: 'Debt Collection', 2: 'Consumer Loan', 3: 'Mortgage'}
print("\n=== Prediction ===")
print(f"Predicted Category for New Complaint: {category_names[predicted_category[0]]}")

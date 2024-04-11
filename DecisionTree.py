import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

try:
    train_df = pd.read_csv('Dataset/train.csv', encoding='ISO-8859-1')
    test_df = pd.read_csv('Dataset/test.csv', encoding='ISO-8859-1')
    success = True
except Exception as e:
    print("Error encountered:", e)
    success = False

if success:
    print("Train dataset head:")
    print(train_df.head())
    print("\nTest dataset head:")
    print(test_df.head())
else:
    print("Failed to load datasets.")



X = train_df.drop(['sentiment'], axis=1)
y = train_df['sentiment']

X = X.select_dtypes(include=[np.number])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

val_accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {val_accuracy}")

depths = range(1, 21)  
accuracy_scores = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    accuracy_scores.append(accuracy)


plt.figure(figsize=(10, 6))
plt.plot(depths, accuracy_scores, marker='o')
plt.title('Decision Tree Model Complexity vs. Validation Accuracy')
plt.xlabel('Max Depth of Tree')
plt.ylabel('Accuracy on Validation Set')
plt.grid(True)
plt.show()

optimal_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
optimal_clf.fit(X_train, y_train)

features = X.columns
importances = optimal_clf.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 8))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
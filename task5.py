import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import graphviz

# Step 1: Load the dataset
df = pd.read_csv('heart.csv')  # Make sure the file is in the same folder

# Step 2: Show basic info about the data
print("First 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Step 3: Split data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Step 4: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Decision Tree Classifier
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

# Step 6: Predict with Decision Tree
y_pred = dtree.predict(X_test)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred))

# Step 7: Visualize the Decision Tree
dot_data = export_graphviz(dtree, out_file=None, 
                           feature_names=X.columns,
                           class_names=["No Disease", "Disease"],
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format='png', cleanup=False)  # Output: decision_tree.png
print("\nDecision Tree saved as 'decision_tree.png'.")

# Step 8: Train Decision Tree with limited depth (to prevent overfitting)
dtree_limited = DecisionTreeClassifier(max_depth=3, random_state=42)
dtree_limited.fit(X_train, y_train)
y_pred_limited = dtree_limited.predict(X_test)
print("Limited Depth Tree Accuracy:", accuracy_score(y_test, y_pred_limited))

# Step 9: Train Random Forest Classifier
rforest = RandomForestClassifier(n_estimators=100, random_state=42)
rforest.fit(X_train, y_train)
y_pred_rf = rforest.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Step 10: Feature Importance from Random Forest
importances = rforest.feature_importances_
features = X.columns

# Plotting Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importances in Random Forest")
plt.tight_layout()
plt.show()

# Step 11: Cross-validation with Random Forest
cv_scores = cross_val_score(rforest, X, y, cv=5)
print("Cross-validation Scores:", cv_scores)
print("Average CV Accuracy:", cv_scores.mean())

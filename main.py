import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# 1. Load the dataset from a CSV file
df = pd.read_csv('lang_scores.csv')

# 2. Data cleaning
print(df.isnull())
print(df.isnull().sum())
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(df.isnull().sum())

# 3. Select the features and target variable
X = df[['Reading', 'Listening', 'Speaking', 'Writing']]
y = df['LangLevel']

# 4. Encode categorical variables as numeric
X = pd.get_dummies(X)

# 5. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Hyperparameter tuning
param_grid = {'max_depth': [3, 4, 5, 6, 7],
              'min_samples_split': [2, 3, 4, 5, 6],
              'min_samples_leaf': [1, 2, 3, 4, 5]}

clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
clf.fit(X_train, y_train)

# pickle.dump(clf, open("model.pkl", "wb"))

# 7. Make predictions on the testing set
y_pred = clf.predict(X_test)

# 8. Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Best parameters:", clf.best_params_)

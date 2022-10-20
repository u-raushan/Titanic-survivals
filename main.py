import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsClassifier

# Read data sets
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
#test_data = test_data[test_data['Fare'].notna()]

# Initialize input features and output
features = ["Pclass", "Sex", "Parch", "SibSp", "Embarked"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

y = train_data["Survived"]

# Initialize classifier and train
model = KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=1)
model.fit(X, y)
predictions = model.predict(X_test)

# Save predictions
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

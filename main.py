import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


# load the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# combine datas for preprocessing
combined_data = pd.concat([train_data.drop('outcome', axis=1), test_data])

# handling missing values
combined_data.replace('?', pd.NA, inplace=True)

# impute missing values (replace NaNs) with the mean for numerical columns
numerical_cols = combined_data.select_dtypes(include='number').columns.tolist()
imputer = SimpleImputer(strategy='mean')
combined_data[numerical_cols] = imputer.fit_transform(combined_data[numerical_cols])

# perform label encoding for categorical columns
encoder = LabelEncoder()
categorical_cols = combined_data.select_dtypes(include='object').columns.tolist()
for col in categorical_cols:
    combined_data[col] = encoder.fit_transform(combined_data[col].astype(str))

# split back into train and test datasets after label encoding
X_train = combined_data[:len(train_data)]
X_test = combined_data[len(train_data):]
y = train_data['outcome']

# standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# apply PCA for dimensionality reduction
pca = PCA(n_components=10)  # Adjust the number of components as needed
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# split the pca-transformed data into train and validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_pca, y, test_size=0.2, random_state=42)

# initialize and fit knn model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_split, y_train_split)

# predictions on the validation set
predictions = knn.predict(X_val)

# accuracy on the validation set
accuracy = accuracy_score(y_val, predictions)
print("Validation set accuracy:", accuracy)

# predictions on the test set
test_predictions = knn.predict(X_test_pca)

submission = pd.DataFrame({'id': test_data['id'], 'outcome': test_predictions})
submission.to_csv('my_submission.csv', index=False)


# hyperparameter tuning for KNN
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}  
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train_split, y_train_split)

best_knn = grid_search.best_estimator_
best_knn.fit(X_train_split, y_train_split)

#predictions on the validation set
best_predictions = best_knn.predict(X_val)
best_accuracy = accuracy_score(y_val, best_predictions)
print("Validation set accuracy after tuning:", best_accuracy)

# predictions on the test set
test_predictions = best_knn.predict(X_test_pca)

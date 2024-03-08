import pandas as pd


file_path = '"C:\Users\kusha\Downloads\ml-assign5\simplified_coffee.csv"'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
data.head()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Convert ratings into two classes based on a threshold
# Assuming ratings >= 92 are high (class 1) and below are low (class 0)
data['class'] = np.where(data['rating'] >= 92, 1, 0)


X = data[['100g_USD']]  # Feature matrix
y = data['class']  # Target variable

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on training and test data
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

# Evaluate confusion matrix and other metrics for both training and test data
confusion_matrix_train = confusion_matrix(y_train, y_pred_train)
confusion_matrix_test = confusion_matrix(y_test, y_pred_test)
report_train = classification_report(y_train, y_pred_train, output_dict=True)
report_test = classification_report(y_test, y_pred_test, output_dict=True)

(confusion_matrix_train, confusion_matrix_test, report_train,report_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


X_reg = data[['rating']]  # Feature matrix for regression
y_reg = data['100g_USD']  # Target variable for regression

# Split the dataset into training and test sets for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# Initialize and train a Linear Regression model
reg = LinearRegression()
reg.fit(X_train_reg, y_train_reg)

# Predict on test data
y_pred_reg = reg.predict(X_test_reg)

# Calculate MSE, RMSE, MAPE, and R^2 scores
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = mean_squared_error(y_test_reg, y_pred_reg, squared=False)
mape = np.mean(np.abs((y_test_reg - y_pred_reg) / y_test_reg)) * 100
r2 = r2_score(y_test_reg, y_pred_reg)

(mse,rmse,mape,r2)

import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)  
X_feature = np.random.uniform(1, 10, 20)
Y_feature = np.random.uniform(1, 10, 20)

# Assign to classes based on a simple rule to have a mix of classes
# For demonstration, let's classify based on whether X+Y is above or below the median
class_threshold = np.median(X_feature + Y_feature)
classes = np.where((X_feature + Y_feature) > class_threshold, 1, 0)  # Class 1 if X+Y is above median, else Class 0

# Plot the training data
plt.figure(figsize=(8, 6))
for i in range(len(classes)):
    if classes[i] == 0:
        plt.scatter(X_feature[i], Y_feature[i], color='blue', label='Class 0' if i == 0 else "")
    else:
        plt.scatter(X_feature[i], Y_feature[i], color='red', label='Class 1' if i == 1 else "")
plt.title('Scatter Plot of Training Data')
plt.xlabel('X Feature')
plt.ylabel('Y Feature')
plt.legend()
plt.show()

from sklearn.neighbors import KNeighborsClassifier


X_test_grid, Y_test_grid = np.meshgrid(np.arange(0, 10.1, 0.1), np.arange(0, 10.1, 0.1))
test_data = np.c_[X_test_grid.ravel(), Y_test_grid.ravel()]

# Training data
training_data = np.column_stack((X_feature, Y_feature))

# kNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(training_data, classes)

# Classify the test set data
test_predictions = knn.predict(test_data)

# Plot the test data output
plt.figure(figsize=(10, 8))
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_predictions, cmap='bwr', alpha=0.5)
plt.title('Scatter Plot of Test Data Classified by kNN (k=3)')
plt.xlabel('X Feature')
plt.ylabel('Y Feature')
plt.show()

# Define a function to plot for different k values
def plot_knn_classification(k_value):
    # kNN classifier with given k
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(training_data, classes)

    # Classify the test set data
    test_predictions = knn.predict(test_data)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(test_data[:, 0], test_data[:, 1], c=test_predictions, cmap='bwr', alpha=0.5)
    plt.title(f'Scatter Plot of Test Data Classified by kNN (k={k_value})')
    plt.xlabel('X Feature')
    plt.ylabel('Y Feature')
    plt.show()

# Plot for various k values
k_values = [1, 5, 10]
for k in k_values:
    plot_knn_classification(k)
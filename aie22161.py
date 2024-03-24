import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Load and preprocess your data
data_path = "C:\\Users\\kusha\\Desktop\\ml-assign7\\simplified_coffee.csv"  
coffee_data = pd.read_csv(data_path)
X = coffee_data.drop(['name', 'rating', 'review_date', 'review'], axis=1)  
y = coffee_data['rating']  #
numerical_cols = ['100g_USD']  
categorical_cols = ['roaster', 'roast', 'loc_country', 'origin']  

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
dt_pipeline = make_pipeline(preprocessor, DecisionTreeClassifier(random_state=42))
dt_pipeline.fit(X_train, y_train)
dt_pred = dt_pipeline.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_pred)}")

# Random Forest
rf_pipeline = make_pipeline(preprocessor, RandomForestClassifier(random_state=42))
rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred)}")

# AdaBoost
ab_pipeline = make_pipeline(preprocessor, AdaBoostClassifier(random_state=42))
ab_pipeline.fit(X_train, y_train)
ab_pred = ab_pipeline.predict(X_test)
print(f"AdaBoost Accuracy: {accuracy_score(y_test, ab_pred)}")

# MLP with RandomizedSearchCV
mlp_pipeline = make_pipeline(preprocessor, MLPClassifier(max_iter=1000, random_state=42))
param_dist_mlp = {
    'mlpclassifier__hidden_layer_sizes': [(50,), (100,)],
    'mlpclassifier__activation': ['tanh', 'relu'],
}

rnd_search_mlp = RandomizedSearchCV(mlp_pipeline, param_distributions=param_dist_mlp)

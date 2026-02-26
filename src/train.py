import pandas as pd 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression 

# Load the dataset
data = pd.read_csv('data/raw.csv')

data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce')
df = data.dropna()

X = df.drop(columns=['Churn', 'customerID'])
y = df['Churn'].map({'Yes': 1, 'No': 0})    

num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[ 
        ("num", numeric_transformer, num_features), 
        ("cat", categorical_transformer, cat_features) 
    ]
)

model = LogisticRegression(max_iter=1000)

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'models/churn_model.pkl')

print("Model training completed and saved to 'models/churn_model.pkl'")

